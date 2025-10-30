/// <reference lib="dom" />
/**
 * @fileoverview Control real time music with text prompts
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import {css, CSSResultGroup, html, LitElement, svg} from 'lit';
import {customElement, property, query, state}from 'lit/decorators.js';
import {classMap} from 'lit/directives/class-map.js';
import {styleMap} from 'lit/directives/style-map.js';
import {map} from 'lit/directives/map.js';

import {
  GoogleGenAI,
  type LiveMusicGenerationConfig,
  type LiveMusicSession,
  type LiveServerMessage,
  Modality,
} from '@google/genai';
import {createBlob, decode, decodeAudioData} from './utils';
import {FFmpeg} from '@ffmpeg/ffmpeg';
import {fetchFile, toBlobURL} from '@ffmpeg/util';
import type {Mutable} from 'utility-types'; // Import for FFmpeg types

const ai = new GoogleGenAI({
  apiKey: process.env.API_KEY,
});
let model = 'music-realtime-fm';

interface Prompt {
  readonly promptId: string;
  readonly color: string;
  text: string;
  weight: number;
}

type PlaybackState = 'stopped' | 'playing' | 'loading' | 'paused';
type RecordingState =
  | 'idle'
  | 'initializing'
  | 'recording'
  | 'processing'
  | 'finished';

/** Throttles a callback to be called at most once per `freq` milliseconds. */
function throttle(func: (...args: unknown[]) => void, delay: number) {
  let lastCall = 0;
  return (...args: unknown[]) => {
    const now = Date.now();
    const timeSinceLastCall = now - lastCall;
    if (timeSinceLastCall >= delay) {
      func(...args);
      lastCall = now;
    }
  };
}

const PROMPT_TEXT_PRESETS = [
  'Bossa Nova',
  'Minimal Techno',
  'Drum and Bass',
  'Post Punk',
  'Shoegaze',
  'Funk',
  'Chiptune',
  'Lush Strings',
  'Sparkling Arpeggios',
  'Staccato Rhythms',
  'Punchy Kick',
  'Dubstep',
  'K Pop',
  'Neo Soul',
  'Trip Hop',
  'Thrash',
];

const COLORS = [
  '#9900ff',
  '#5200ff',
  '#ff25f6',
  '#2af6de',
  '#ffdd28',
  '#3dffab',
  '#d8ff3e',
  '#d9b2ff',
];

function getUnusedRandomColor(usedColors: string[]): string {
  const availableColors = COLORS.filter((c) => !usedColors.includes(c));
  if (availableColors.length === 0) {
    // If no available colors, pick a random one from the original list.
    return COLORS[Math.floor(Math.random() * COLORS.length)];
  }
  return availableColors[Math.floor(Math.random() * availableColors.length)];
}

/** Converts a hex color string to an RGBA string. */
function hexToRgbA(hex: string, alpha: number): string {
  let c: string | string[] | number;
  if (/^#([A-Fa-f0-9]{3}){1,2}$/.test(hex)) {
    c = hex.substring(1).split('');
    if (c.length === 3) {
      c = [c[0], c[0], c[1], c[1], c[2], c[2]];
    }
    c = '0x' + c.join('');
    return 'rgba(' + [(Number(c) >> 16) & 255, (Number(c) >> 8) & 255, Number(c) & 255].join(',') + ',' + alpha + ')';
  }
  // Fallback for invalid hex input
  console.warn(`Invalid hex color: ${hex}. Falling back to black with alpha.`);
  return `rgba(0,0,0,${alpha})`;
}


// WeightSlider component
// -----------------------------------------------------------------------------
/** A slider for adjusting and visualizing prompt weight. */
@customElement('weight-slider')
class WeightSlider extends LitElement {
  static override styles = css`
    :host {
      cursor: ns-resize;
      position: relative;
      height: 100%;
      display: flex;
      justify-content: center;
      flex-direction: column;
      align-items: center;
      padding: 5px;
    }
    .scroll-container {
      width: 100%;
      flex-grow: 1;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
    }
    .value-display {
      font-size: 1.3vmin;
      color: #ccc;
      margin: 0.5vmin 0;
      user-select: none;
      text-align: center;
    }
    .slider-container {
      position: relative;
      width: 10px;
      height: 100%;
      background-color: #0009;
      border-radius: 4px;
    }
    #thumb {
      position: absolute;
      bottom: 0;
      left: 0;
      width: 100%;
      border-radius: 4px;
      box-shadow: 0 0 3px rgba(0, 0, 0, 0.7);
    }
  `;

  @property({type: Number}) value = 0; // Range 0-2
  @property({type: String}) color = '#000';

  @query('.scroll-container') private scrollContainer!: HTMLDivElement;

  private dragStartPos = 0;
  private dragStartValue = 0;
  private containerBounds: DOMRect | null = null;

  private handlePointerDown(e: PointerEvent) {
    e.preventDefault();
    this.containerBounds = this.scrollContainer.getBoundingClientRect();
    this.dragStartPos = e.clientY;
    this.dragStartValue = this.value;
    document.body.classList.add('dragging');
    window.addEventListener('pointermove', this.handlePointerMove);
    window.addEventListener('touchmove', this.handleTouchMove, {
      passive: false,
    });
    window.addEventListener('pointerup', this.handlePointerUp, {once: true});
    this.updateValueFromPosition(e.clientY);
  }

  private handlePointerMove(e: PointerEvent) {
    this.updateValueFromPosition(e.clientY);
  }

  private handleTouchMove(e: TouchEvent) {
    e.preventDefault();
    this.updateValueFromPosition(e.touches[0].clientY);
  }

  private handlePointerUp(e: PointerEvent) {
    window.removeEventListener('pointermove', this.handlePointerMove);
    document.body.classList.remove('dragging');
    window.removeEventListener('touchmove', this.handleTouchMove);
    this.containerBounds = null;
  }

  private updateValueFromPosition(clientY: number) {
    if (!this.containerBounds) {
      return;
    }
    const deltaY = this.dragStartPos - clientY;
    const dragRatio = deltaY / this.containerBounds.height;
    let newValue = this.dragStartValue + dragRatio * 2; // Scale sensitivity
    newValue = Math.max(0, Math.min(2, newValue)); // Clamp between 0 and 2
    this.value = parseFloat(newValue.toFixed(2));
    this.dispatchEvent(new CustomEvent('value-change', {detail: this.value}));
  }

  override render() {
    const thumbHeight = `${Math.max(10, this.value / 2 * 100)}%`; // Min height for visibility, max 100%
    const thumbStyle = styleMap({
      height: thumbHeight,
      backgroundColor: this.color,
    });
    return html`
      <div
        class="scroll-container"
        @pointerdown=${this.handlePointerDown}
        @touchstart=${this.handlePointerDown}
      >
        <div class="slider-container">
          <div id="thumb" style=${thumbStyle}></div>
        </div>
        <div class="value-display">${this.value.toFixed(2)}</div>
      </div>
    `;
  }
}

// Main App component
// -----------------------------------------------------------------------------
@customElement('music-live-stream-app')
export class MusicLiveStreamApp extends LitElement {
  static override styles: CSSResultGroup = css`
    :host {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: flex-start;
      min-height: 100vh;
      background-color: #1a1a1a;
      color: #e0e0e0;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      font-size: 1.8vmin;
      box-sizing: border-box;
      padding: 2vmin;
    }

    .main-container {
      display: flex;
      flex-direction: column;
      gap: 20px;
      width: 100%;
      max-width: 900px;
      padding: 20px;
      border-radius: 12px;
      background-color: #282828;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
      min-height: calc(100vh - 4vmin); /* Adjust for padding */
      box-sizing: border-box;
    }

    header {
      text-align: center;
      margin-bottom: 20px;
    }

    header h1 {
      font-size: 4.5vmin;
      color: #3dffab;
      margin-bottom: 5px;
      text-shadow: 0 0 10px rgba(61, 255, 171, 0.6);
    }

    header p {
      font-size: 1.8vmin;
      color: #aaa;
    }

    section {
      margin-bottom: 15px;
    }

    .prompt-input-area {
      display: flex;
      gap: 10px;
    }

    #promptInput {
      flex-grow: 1;
      padding: 12px 15px;
      border: 1px solid #555;
      border-radius: 8px;
      background-color: #3a3a3a;
      color: #e0e0e0;
      font-size: 1.8vmin;
      transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }

    #promptInput::placeholder {
      color: #888;
    }

    #promptInput:focus {
      border-color: #3dffab;
      box-shadow: 0 0 8px rgba(61, 255, 171, 0.4);
      outline: none;
    }

    button {
      padding: 12px 20px;
      border: none;
      border-radius: 8px;
      background-color: #3dffab;
      color: #1a1a1a;
      font-weight: bold;
      font-size: 1.8vmin;
      cursor: pointer;
      transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }

    button:hover:not(:disabled) {
      background-color: #2af6de;
      transform: translateY(-2px);
      box-shadow: 0 4px 15px rgba(42, 246, 222, 0.4);
    }

    button:active:not(:disabled) {
      transform: translateY(0);
      box-shadow: none;
    }

    button:disabled {
      background-color: #555;
      color: #bbb;
      cursor: not-allowed;
      opacity: 0.6;
    }

    .prompt-presets {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      justify-content: center;
      margin-top: -5px; /* Adjust for gap from input area */
    }

    .preset-button {
      background-color: #444;
      color: #e0e0e0;
      padding: 8px 15px;
      font-size: 1.6vmin;
      border-radius: 20px;
      white-space: nowrap;
    }

    .preset-button:hover:not(:disabled) {
      background-color: #555;
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }

    .prompts-list-area {
      display: flex;
      flex-direction: column;
      gap: 10px;
      max-height: 300px;
      overflow-y: auto;
      padding-right: 5px; /* For scrollbar */
    }

    .prompts-list-area::-webkit-scrollbar {
      width: 8px;
    }

    .prompts-list-area::-webkit-scrollbar-track {
      background: #3a3a3a;
      border-radius: 10px;
    }

    .prompts-list-area::-webkit-scrollbar-thumb {
      background: #555;
      border-radius: 10px;
    }

    .prompts-list-area::-webkit-scrollbar-thumb:hover {
      background: #777;
    }

    .prompt-item {
      display: flex;
      align-items: stretch; /* Make children stretch to full height */
      gap: 10px;
      padding: 8px;
      border-left: 5px solid; /* Placeholder, color set by JS */
      border-radius: 8px;
      background-color: rgba(0, 0, 0, 0.2); /* Placeholder, color set by JS */
      min-height: 60px;
    }

    .prompt-text-wrapper {
      flex-grow: 1;
      display: flex;
      align-items: center;
      justify-content: space-between;
      color: #e0e0e0;
      font-size: 1.8vmin;
      background-color: #3a3a3a;
      border-radius: 6px;
      padding: 0 10px;
    }

    .prompt-text {
      flex-grow: 1;
    }

    .remove-prompt-button {
      background: none;
      border: none;
      color: #ff25f6;
      cursor: pointer;
      font-size: 2vmin;
      padding: 0;
      width: 30px;
      height: 30px;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .remove-prompt-button:hover {
      color: #ffdd28;
      transform: scale(1.1);
      background-color: rgba(255, 37, 246, 0.1);
      border-radius: 50%;
    }
    .remove-prompt-button svg {
      width: 1em;
      height: 1em;
    }

    weight-slider {
      width: 40px; /* Fixed width for slider */
      flex-shrink: 0;
    }

    .controls-area {
      display: flex;
      gap: 15px;
      justify-content: center;
      align-items: center;
      padding-top: 15px;
      border-top: 1px solid #444;
      margin-top: auto; /* Push controls to the bottom */
    }

    .play-button, .record-button {
      min-width: 120px;
    }

    .play-button {
      background-color: #3dffab;
    }
    .play-button:hover:not(:disabled) {
      background-color: #2af6de;
    }

    .record-button {
      background-color: #ff25f6;
    }

    .record-button:hover:not(:disabled) {
      background-color: #d9b2ff;
    }

    .record-button.recording-active {
      background-color: #e74c3c; /* Red for active recording */
      animation: pulse 1.5s infinite;
    }

    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0.7); }
      70% { box-shadow: 0 0 0 10px rgba(231, 76, 60, 0); }
      100% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0); }
    }

    .spinner {
      border: 4px solid rgba(255, 255, 255, 0.3);
      border-top: 4px solid #fff;
      border-radius: 50%;
      width: 16px;
      height: 16px;
      animation: spinner-animation 1s linear infinite;
      display: inline-block;
      vertical-align: middle;
      margin-right: 8px;
    }
    @keyframes spinner-animation {
      0% {
        transform: rotate(0deg);
      }
      100% {
        transform: rotate(360deg);
      }
    }

    .download-link {
      display: block;
      margin-top: 20px;
      text-align: center;
      color: #ffdd28;
      text-decoration: none;
      font-size: 1.8vmin;
      padding: 10px 20px;
      border: 1px solid #ffdd28;
      border-radius: 8px;
      transition: background-color 0.3s ease, color 0.3s ease;
    }

    .download-link:hover {
      background-color: #ffdd28;
      color: #1a1a1a;
      box-shadow: 0 0 15px rgba(255, 221, 40, 0.5);
    }

    /* Volume Slider Styles */
    .volume-control {
      display: flex;
      align-items: center;
      gap: 10px;
      color: #ccc;
      font-size: 1.6vmin;
      flex-grow: 1; /* Allow it to take available space */
      justify-content: center; /* Center content within its flex item */
    }

    .volume-control label {
      white-space: nowrap;
    }

    #volume-slider {
      -webkit-appearance: none;
      width: 150px; /* Or a percentage like 70% for responsiveness */
      height: 8px;
      background: #333;
      border-radius: 4px;
      outline: none;
      opacity: 0.7;
      transition: opacity 0.2s;
    }

    #volume-slider:hover {
      opacity: 1;
    }

    #volume-slider::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background: #4CAF50; /* A pleasant green */
      cursor: pointer;
      box-shadow: 0 0 5px rgba(0, 0, 0, 0.5);
    }

    #volume-slider::-moz-range-thumb {
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background: #4CAF50;
      cursor: pointer;
      box-shadow: 0 0 5px rgba(0, 0, 0, 0.5);
    }
  `;

  @property({type: String}) currentPromptInput: string = '';
  @state() prompts: Prompt[] = [];
  @state() playbackState: PlaybackState = 'stopped';
  @state() recordingState: RecordingState = 'idle';
  @state() recordedAudioUrl: string | null = null;
  @state() private outputVolume: number = 0.5; // New state for volume

  // Audio Contexts and Nodes
  @state() private inputAudioContext: AudioContext | null = null;
  @state() private outputAudioContext: AudioContext | null = null;
  @state() private outputGainNode: GainNode | null = null;
  private nextStartTime = 0;
  private sources = new Set<AudioBufferSourceNode>();
  private mediaStream: MediaStream | null = null;
  private scriptProcessor: ScriptProcessorNode | null = null;
  private sessionPromise: Promise<LiveMusicSession> | null = null;

  private mediaRecorder: MediaRecorder | null = null;
  private recordedChunks: Blob[] = [];
  private ffmpeg: FFmpeg | null = null;

  connectedCallback() {
    super.connectedCallback();
    this.initializeAudioContexts();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    this.cleanupAudioResources();
  }

  private initializeAudioContexts() {
    this.inputAudioContext = new (window.AudioContext || window.webkitAudioContext)({sampleRate: 16000});
    this.outputAudioContext = new (window.AudioContext || window.webkitAudioContext)({sampleRate: 24000});
    this.outputGainNode = this.outputAudioContext.createGain();
    this.outputGainNode.connect(this.outputAudioContext.destination);
    this.outputGainNode.gain.value = this.outputVolume; // Set initial volume
  }

  private cleanupAudioResources() {
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }
    if (this.scriptProcessor) {
      this.scriptProcessor.disconnect();
      this.scriptProcessor.onaudioprocess = null;
      this.scriptProcessor = null;
    }
    if (this.inputAudioContext) {
      this.inputAudioContext.close().catch(console.error);
    }
    if (this.outputAudioContext) {
      this.outputAudioContext.close().catch(console.error);
    }
    if (this.sessionPromise) {
        this.sessionPromise.then(session => session.close()).catch(console.error);
        this.sessionPromise = null;
    }
    for (const source of this.sources.values()) {
        source.stop();
    }
    this.sources.clear();
    this.nextStartTime = 0;
  }

  private get currentMusicConfig(): LiveMusicGenerationConfig {
    if (this.prompts.length === 0) {
      return {weightedPrompts: [{text: 'ambient pads', weight: 0.7}]};
    }
    return {
      weightedPrompts: this.prompts.map((p) => ({
        text: p.text,
        weight: p.weight,
      })),
    };
  }

  private addPrompt() {
    const text = this.currentPromptInput.trim();
    if (text === '') {
      return;
    }
    const usedColors = this.prompts.map((p) => p.color);
    const newPrompt: Prompt = {
      promptId: `prompt-${Date.now()}`,
      text: text,
      weight: 1.0,
      color: getUnusedRandomColor(usedColors),
    };
    this.prompts = [...this.prompts, newPrompt];
    this.currentPromptInput = '';
    this.sendPromptUpdate();
  }

  private addPresetPrompt(presetText: string) {
    const usedColors = this.prompts.map((p) => p.color);
    const newPrompt: Prompt = {
      promptId: `prompt-${Date.now()}`,
      text: presetText,
      weight: 1.0,
      color: getUnusedRandomColor(usedColors),
    };
    this.prompts = [...this.prompts, newPrompt];
    this.sendPromptUpdate();
  }

  private removePrompt(promptId: string) {
    this.prompts = this.prompts.filter((p) => p.promptId !== promptId);
    this.sendPromptUpdate();
  }

  private updatePromptWeight(promptId: string, newWeight: number) {
    this.prompts = this.prompts.map((p) =>
      p.promptId === promptId ? {...p, weight: newWeight} : p,
    );
    this.sendPromptUpdate();
  }

  private handlePromptInputKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter') {
      this.addPrompt();
    }
  }

  private sendPromptUpdate = throttle(() => {
    // This function will be throttled, preventing rapid updates.
    if (this.sessionPromise) {
      this.sessionPromise.then((session) => {
        session.sendRealtimeInput({musicGenerationConfig: this.currentMusicConfig});
      });
    }
    this.requestUpdate();
  }, 100); // Throttles to 100ms

  private async connectLiveSession() {
    if (this.playbackState === 'playing' || this.playbackState === 'loading') {
      return;
    }
    this.playbackState = 'loading';
    this.requestUpdate();

    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({audio: true});

      if (!this.inputAudioContext || !this.outputAudioContext || !this.outputGainNode) {
        console.error('Audio contexts or gain node not initialized. Re-initializing.');
        this.initializeAudioContexts();
        if (!this.inputAudioContext || !this.outputAudioContext || !this.outputGainNode) {
            alert('Failed to initialize audio contexts. Please check browser settings.');
            this.playbackState = 'stopped';
            this.requestUpdate();
            return;
        }
      }

      this.sessionPromise = ai.live.connect({
        model: model,
        config: {
          responseModalities: [Modality.AUDIO],
          musicGenerationConfig: this.currentMusicConfig,
        },
        callbacks: {
          onopen: () => {
            if (!this.inputAudioContext || !this.mediaStream) {
              console.error('Input audio context or media stream not available on open.');
              return;
            }
            const source = this.inputAudioContext.createMediaStreamSource(this.mediaStream!);
            // Cast to Mutable<ScriptProcessorNode> as ScriptProcessorNode is being used.
            this.scriptProcessor = this.inputAudioContext.createScriptProcessor(4096, 1, 1) as Mutable<ScriptProcessorNode>;
            this.scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
              const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
              const pcmBlob = createBlob(inputData);
              this.sessionPromise!.then((session) => {
                session.sendRealtimeInput({media: pcmBlob});
              }).catch(e => console.error("Error sending realtime input:", e));
            };
            source.connect(this.scriptProcessor);
            this.scriptProcessor.connect(this.inputAudioContext.destination);

            this.playbackState = 'playing';
            this.requestUpdate();
          },
          onmessage: async (message: LiveServerMessage) => {
            const audioPart = message.serverContent?.modelTurn?.parts?.[0];
            if (audioPart?.inlineData?.data && this.outputAudioContext && this.outputGainNode) {
              this.nextStartTime = Math.max(
                this.nextStartTime,
                this.outputAudioContext.currentTime,
              );
              const audioBuffer = await decodeAudioData(
                decode(audioPart.inlineData.data),
                this.outputAudioContext,
                24000,
                1,
              );
              const source = this.outputAudioContext.createBufferSource();
              source.buffer = audioBuffer;
              source.connect(this.outputGainNode);
              source.addEventListener('ended', () => {
                this.sources.delete(source);
              });

              source.start(this.nextStartTime);
              this.nextStartTime = this.nextStartTime + audioBuffer.duration;
              this.sources.add(source);
            }

            const interrupted = message.serverContent?.interrupted;
            if (interrupted) {
              for (const source of this.sources.values()) {
                source.stop();
                this.sources.delete(source);
              }
              this.nextStartTime = 0;
            }
            this.requestUpdate();
          },
          onerror: (e: Event) => {
            console.error('Live session error:', e);
            this.playbackState = 'stopped';
            this.requestUpdate();
            alert('Music stream encountered an error.');
            this.cleanupAudioResources(); // Clean up on error
          },
          onclose: (e: CloseEvent) => {
            console.debug('Live session closed:', e);
            this.playbackState = 'stopped';
            this.requestUpdate();
            this.cleanupAudioResources(); // Clean up on close
          },
        },
      });
    } catch (error) {
      console.error('Error connecting live session:', error);
      this.playbackState = 'stopped';
      this.requestUpdate();
      alert('Failed to connect to the music stream. Please ensure microphone access is granted.');
      this.cleanupAudioResources(); // Clean up on connection error
    }
  }

  private async stopPlayback() {
    if (this.sessionPromise) {
      const session = await this.sessionPromise;
      session.close();
      this.sessionPromise = null;
    }
    for (const source of this.sources.values()) {
      source.stop();
    }
    this.sources.clear();
    this.nextStartTime = 0;
    this.playbackState = 'stopped';
    this.requestUpdate();
    // Also ensure microphone stream and processor are stopped
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }
    if (this.scriptProcessor) {
      this.scriptProcessor.disconnect();
      this.scriptProcessor.onaudioprocess = null;
      this.scriptProcessor = null;
    }
  }

  private async initializeFfmpeg() {
    if (this.ffmpeg) {
      return;
    }
    this.ffmpeg = new FFmpeg();
    this.ffmpeg.on('log', ({message}) => console.log(`FFmpeg: ${message}`));
    this.ffmpeg.on('progress', ({progress, time}) =>
      console.log(`FFmpeg progress: ${progress * 100}% (time: ${time})`),
    );
    const baseURL = 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/esm';
    await this.ffmpeg.load({
      coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, 'text/javascript'),
      wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
    });
  }

  private async toggleRecording() {
    if (this.recordingState === 'recording') {
      this.stopRecording();
    } else if (this.playbackState === 'playing') {
      await this.startRecording();
    }
  }

  private async startRecording() {
    if (this.recordingState !== 'idle' && this.recordingState !== 'finished') {
      return;
    }

    this.recordedChunks = [];
    this.recordedAudioUrl = null;
    this.recordingState = 'initializing';
    this.requestUpdate();

    try {
      const options = {mimeType: 'audio/webm'};
      this.mediaRecorder = new MediaRecorder(this.outputAudioContext!.createMediaStreamDestination().stream, options);

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.recordedChunks.push(event.data);
        }
      };

      this.mediaRecorder.onstop = async () => {
        this.recordingState = 'processing';
        this.requestUpdate();

        const webmBlob = new Blob(this.recordedChunks, {type: 'audio/webm'});
        const arrayBuffer = await webmBlob.arrayBuffer();
        const inputFileName = 'input.webm';
        const outputFileName = 'auralspirit_recording.mp3';

        await this.initializeFfmpeg();

        await this.ffmpeg!.writeFile(inputFileName, new Uint8Array(arrayBuffer));
        await this.ffmpeg!.exec([
          '-i',
          inputFileName,
          '-f',
          'mp3',
          '-q:a',
          '2', // Variable bitrate (VBR) quality level (0-9, where 0 is highest quality)
          outputFileName,
        ]);

        const data = await this.ffmpeg!.readFile(outputFileName);
        const mp3Blob = new Blob([data as Uint8Array], {type: 'audio/mp3'});
        this.recordedAudioUrl = URL.createObjectURL(mp3Blob);
        this.recordingState = 'finished';
        this.requestUpdate();
      };

      this.mediaRecorder.start();
      this.recordingState = 'recording';
      this.requestUpdate();
      console.log('Recording started');
    } catch (error) {
      console.error('Error starting recording:', error);
      this.recordingState = 'idle';
      this.requestUpdate();
      alert('Failed to start recording. Please ensure browser permissions.');
    }
  }

  private stopRecording() {
    if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
      this.mediaRecorder.stop();
      console.log('Recording stopped');
    }
  }

  private handleVolumeChange(e: Event) {
    const newVolume = (e.target as HTMLInputElement).valueAsNumber;
    this.outputVolume = newVolume;
    if (this.outputGainNode) {
      this.outputGainNode.gain.value = newVolume;
    }
  }

  override render() {
    const isPlaying = this.playbackState === 'playing';
    const isLoading = this.playbackState === 'loading';
    const isStopped = this.playbackState === 'stopped';
    const isPaused = this.playbackState === 'paused';
    const isRecording = this.recordingState === 'recording';

    const recordingClasses = classMap({
      'recording-active': isRecording,
    });

    return html`
      <div class="main-container">
        <header>
          <h1>AURALSPIRIT</h1>
          <p>Steer a continuous stream of music with text prompts</p>
        </header>

        <section class="prompt-input-area">
          <input
            type="text"
            id="promptInput"
            @keydown=${this.handlePromptInputKeyDown}
            placeholder="Type your music prompt here (e.g., 'chill lo-fi beats')"
            .value=${this.currentPromptInput}
            @input=${(e: Event) => {
              this.currentPromptInput = (e.target as HTMLInputElement).value;
            }}
          />
          <button @click=${this.addPrompt}>Add Prompt</button>
        </section>

        <section class="prompts-list-area">
          ${map(
            this.prompts,
            (prompt) => html`
              <div
                class="prompt-item"
                style=${styleMap({
                  borderColor: prompt.color,
                  backgroundColor: hexToRgbA(prompt.color, 0.1),
                })}
              >
                <div class="prompt-text-wrapper">
                  <span class="prompt-text">${prompt.text}</span>
                  <button
                    class="remove-prompt-button"
                    @click=${() => this.removePrompt(prompt.promptId)}
                  >
                    ${svg`<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M6 18L18 6M6 6L18 18" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>`}
                  </button>
                </div>
                <weight-slider
                  .value=${prompt.weight}
                  .color=${prompt.color}
                  @value-change=${(e: CustomEvent<number>) =>
                    this.updatePromptWeight(prompt.promptId, e.detail)}
                ></weight-slider>
              </div>
            `,
          )}
        </section>

        <section class="prompt-presets">
          ${map(
            PROMPT_TEXT_PRESETS,
            (preset) => html`
              <button class="preset-button" @click=${() => this.addPresetPrompt(preset)}>
                ${preset}
              </button>
            `,
          )}
        </section>

        <section class="controls-area">
          <!-- Play/Stop Button -->
          <button
            class="play-button"
            @click=${isPlaying || isLoading ? this.stopPlayback : this.connectLiveSession}
            ?disabled=${isLoading}
          >
            ${isLoading
              ? html`
                  <div class="spinner"></div>
                  Loading...
                `
              : isPlaying
                ? html`
                    ${svg`<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M4 4H8V20H4V4ZM16 4H20V20H16V4Z" fill="currentColor"/>
                    </svg>`}
                    Stop
                  `
                : html`
                    ${svg`<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M8 5V19L19 12L8 5Z" fill="currentColor"/>
                    </svg>`}
                    Play
                  `}
          </button>

          <!-- Volume Slider -->
          <div class="volume-control">
            <label for="volume-slider">Volume</label>
            <input
              id="volume-slider"
              type="range"
              min="0"
              max="1"
              step="0.01"
              .value=${this.outputVolume}
              @input=${this.handleVolumeChange}
            />
          </div>

          <!-- Record Button -->
          <button
            class="record-button ${recordingClasses}"
            @click=${this.toggleRecording}
            ?disabled=${!isPlaying || isLoading || this.recordingState === 'processing'}
          >
            ${this.recordingState === 'recording'
              ? html`
                  ${svg`<svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="8"/></svg>`}
                  Recording...
                `
              : this.recordingState === 'processing'
                ? html`
                    <div class="spinner"></div>
                    Processing...
                  `
                : html`
                    ${svg`<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <circle cx="12" cy="12" r="8" stroke="currentColor" stroke-width="2"/>
                    </svg>`}
                    Record
                  `}
          </button>
        </section>

        <!-- Recording download link -->
        ${this.recordingState === 'finished' && this.recordedAudioUrl
          ? html`
              <a href=${this.recordedAudioUrl} download="auralspirit_recording.mp3" class="download-link">
                Download Recording
              </a>
            `
          : ''}
      </div>
    `;
  }
}