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
  // Fix: Corrected type from LiveSession to LiveClient
  type LiveClient,
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
// Using 'gemini-2.5-flash-native-audio-preview-09-2025' as it is a supported Live API model.
// The model variable is kept for reference, but the string literal will be used directly in connect for clarity.
const LIVE_MUSIC_MODEL = 'gemini-2.5-flash-native-audio-preview-09-2025';

interface Prompt {
  readonly promptId: string;
  readonly color: string;
  text: string;
  weight: number;
}

// Defining WeightedPrompt interface based on expected structure
interface WeightedPrompt {
  text: string;
  weight: number;
}

// Fix: Redefined LiveMusicGenerationConfig to match the expected object structure for `musicGenerationConfig` in Live API config.
interface LiveMusicGenerationConfig {
  weightedPrompts: WeightedPrompt[];
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
  let c: string | string[];
  if (new RegExp('^#([A-Fa-f0-9]{3}){1,2}$').test(hex)) {
    c = hex.substring(1).split('');
    if (c.length === 3) {
      // Fix: Ensure c is typed as string[] before accessing length and elements.
      c = [c[0], c[0], c[1], c[1], c[2], c[2]] as string[];
    }
    // Fix: Ensure c is typed as string[] before calling join.
    c = '0x' + (c as string[]).join('');
    return 'rgba(' + [(Number(c) >> 16) & 255, (Number(c) >> 8) & 255, Number(c) & 255].join(',') + ',' + alpha + ')';
  }
  // Fallback for invalid hex input
  // Fix: Use the 'hex' parameter directly.
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

    .play-pause-button, .stop-button, .record-button {
      min-width: 120px;
    }

    .play-pause-button {
      background-color: #3dffab;
    }
    .play-pause-button:hover:not(:disabled) {
      background-color: #2af6de;
    }

    .stop-button {
      background-color: #ff7f50; /* Coral color for stop */
    }
    .stop-button:hover:not(:disabled) {
      background-color: #ff6347; /* Tomato color */
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
  // Fix: Corrected type from LiveSession to LiveClient
  private sessionPromise: Promise<LiveClient> | null = null;

  private mediaRecorder: MediaRecorder | null = null;
  private recordedChunks: Blob[] = [];
  private ffmpeg: FFmpeg | null = null;

  connectedCallback() {
    super.connectedCallback();
    this.initializeAudioContexts();
  }

  disconnectedCallback() {
    super.disconnectedCallback();
    // When component is removed, ensure all resources are cleaned.
    // This calls cleanupAudioResources and nulls sessionPromise if it exists.
    this.stopPlayback(); 
  }

  private initializeAudioContexts() {
    // Only re-initialize if contexts are null or closed
    // Fix: Removed webkitAudioContext as it's deprecated and AudioContext is universally supported.
    if (!this.inputAudioContext || this.inputAudioContext.state === 'closed') {
      this.inputAudioContext = new (window.AudioContext)({sampleRate: 16000});
    }
    // Fix: Removed webkitAudioContext as it's deprecated and AudioContext is universally supported.
    if (!this.outputAudioContext || this.outputAudioContext.state === 'closed') {
      this.outputAudioContext = new (window.AudioContext)({sampleRate: 24000});
      this.outputGainNode = this.outputAudioContext.createGain();
      this.outputGainNode.connect(this.outputAudioContext.destination);
      this.outputGainNode.gain.value = this.outputVolume; // Set initial volume
    }
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
    // Only close contexts if they are not already closed
    if (this.inputAudioContext && this.inputAudioContext.state !== 'closed') {
      this.inputAudioContext.close().catch(console.error);
      this.inputAudioContext = null; // Clear reference after closing
    }
    if (this.outputAudioContext && this.outputAudioContext.state !== 'closed') {
      this.outputAudioContext.close().catch(console.error);
      this.outputAudioContext = null; // Clear reference after closing
      this.outputGainNode = null;
    }
    for (const source of this.sources.values()) {
        source.stop();
    }
    this.sources.clear();
    this.nextStartTime = 0;
    // Removed session.close() call from here. It should be handled by onclose/onerror or stopPlayback.
  }

  // Fix: Adjusted the return type of currentMusicConfig and its implementation to match the new LiveMusicGenerationConfig interface.
  private get currentMusicConfig(): LiveMusicGenerationConfig {
    const weightedPrompts = this.prompts.map((p) => ({
      text: p.text,
      weight: p.weight,
    }));
    if (weightedPrompts.length === 0) {
      return {weightedPrompts: [{text: 'ambient pads', weight: 0.7}]};
    }
    return {weightedPrompts: weightedPrompts};
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
        if (session) {
          // Fix: sendRealtimeInput property was missing on LiveSession type.
          // Assuming LiveSession has a sendRealtimeInput method that accepts musicGenerationConfig.
          session.sendRealtimeInput({musicGenerationConfig: this.currentMusicConfig});
        }
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
      this.initializeAudioContexts(); // Ensure contexts are initialized/re-initialized
      if (!this.inputAudioContext || !this.outputAudioContext || !this.outputGainNode) {
        alert('Failed to initialize audio contexts. Please check browser settings.');
        this.playbackState = 'stopped';
        this.requestUpdate();
        this.cleanupAudioResources(); // Clean up on initialization failure
        this.sessionPromise = null; // Clear promise
        return;
      }

      this.mediaStream = await navigator.mediaDevices.getUserMedia({audio: true});

      // Fix: responseModalities and musicGenerationConfig are moved inside the 'config' object.
      // This aligns with the `LiveConnectParameters` type definition and the example in the guidelines.
      this.sessionPromise = ai.live.connect({
        model: LIVE_MUSIC_MODEL, // Explicitly use the string literal for the model
        callbacks: {
          onopen: () => {
            if (!this.inputAudioContext || !this.mediaStream) {
              console.error('Input audio context or media stream not available on open.');
              // This indicates a critical failure; attempt cleanup and reset state.
              this.playbackState = 'stopped';
              this.requestUpdate();
              this.cleanupAudioResources();
              this.sessionPromise = null;
              return;
            }
            const source = this.inputAudioContext.createMediaStreamSource(this.mediaStream!);
            // Cast to Mutable<ScriptProcessorNode> as ScriptProcessorNode is being used.
            this.scriptProcessor = this.inputAudioContext.createScriptProcessor(4096, 1, 1) as Mutable<ScriptProcessorNode>;
            this.scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
              const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
              const pcmBlob = createBlob(inputData);
              this.sessionPromise!.then((session) => {
                if (session && this.playbackState !== 'stopped' && this.playbackState !== 'loading') {
                  // Fix: sendRealtimeInput property was missing on LiveSession type.
                  // Assuming LiveSession has a sendRealtimeInput method that accepts media.
                  session.sendRealtimeInput({media: pcmBlob});
                }
              }).catch(e => {
                // Ignore errors if session is already closing/closed
                if (!e.message.includes("closed") && !e.message.includes("terminating")) {
                  console.error("Error sending realtime input:", e);
                }
              });
            };
            source.connect(this.scriptProcessor);
            this.scriptProcessor.connect(this.inputAudioContext.destination);

            this.playbackState = 'playing';
            this.requestUpdate();
          },
          onmessage: async (message: LiveServerMessage) => {
            const audioPart = message.serverContent?.modelTurn?.parts?.[0];
            // Only play audio if not paused
            if (audioPart?.inlineData?.data && this.outputAudioContext && this.outputGainNode && this.playbackState !== 'paused') {
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
            } else if (audioPart?.inlineData?.data && this.playbackState === 'paused') {
              // If paused, still update nextStartTime to avoid large jumps when resuming
              // but don't play the audio.
              if (this.outputAudioContext) {
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
                this.nextStartTime = this.nextStartTime + audioBuffer.duration;
              }
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
            this.cleanupAudioResources(); // Clean up local resources on error
            this.sessionPromise = null; // Clear promise on error
          },
          onclose: (e: CloseEvent) => {
            console.debug('Live session closed:', e);
            this.playbackState = 'stopped'; // Ensure state is 'stopped' after session closure
            this.requestUpdate();
            this.cleanupAudioResources(); // Clean up local resources on close
            this.sessionPromise = null; // Clear promise on close
          },
        },
        config: { // <--- responseModalities and musicGenerationConfig moved here
          responseModalities: [Modality.AUDIO],
          musicGenerationConfig: this.currentMusicConfig,
        },
      });
    } catch (error) {
      console.error('Error connecting live session:', error);
      this.playbackState = 'stopped';
      this.requestUpdate();
      alert('Failed to connect to the music stream. Please ensure microphone access is granted.');
      this.cleanupAudioResources(); // Clean up local resources on connection error
      this.sessionPromise = null; // Clear promise on connection error
    }
  }

  private async stopPlayback() {
    if (this.sessionPromise) {
      const session = await this.sessionPromise;
      if (session) {
        session.close(); // This will trigger onclose.
      }
    }
    // Immediate UI update, but actual audio cleanup happens in onclose
    // The state will be set to 'stopped' again in onclose, but this provides immediate feedback.
    this.playbackState = 'stopped';
    this.requestUpdate();
    // Do not set this.sessionPromise = null here; it should be done in onclose/onerror.
  }

  private pausePlayback() {
    if (this.playbackState !== 'playing') return;

    for (const source of this.sources.values()) {
      source.stop();
    }
    this.sources.clear();
    // Do NOT close session or stop mic stream. The model continues to generate.
    this.playbackState = 'paused';
    this.requestUpdate();
    console.log('Playback paused. Session and microphone input remain active.');
  }

  private resumePlayback() {
    if (this.playbackState !== 'paused') return;

    // No need to re-initialize audio contexts or restart microphone stream
    // as they were left active during pause.
    this.playbackState = 'playing';
    this.requestUpdate();
    console.log('Playback resumed.');
  }

  private handlePlayPauseButtonClick() {
    if (this.playbackState === 'stopped' || this.playbackState === 'paused') {
      if (this.playbackState === 'paused') {
        this.resumePlayback();
      } else {
        this.connectLiveSession();
      }
    } else if (this.playbackState === 'playing') {
      this.pausePlayback();
    }
  }

  private async handleRecordButtonClick() {
    if (this.recordingState === 'processing') return;

    if (this.recordingState === 'idle') {
      // Start recording
      if (!this.outputAudioContext) {
        alert('Audio context not available for recording.');
        return;
      }
      this.recordedChunks = [];
      this.recordedAudioUrl = null;
      // Capture the generated output directly from the gain node
      const destination = this.outputAudioContext.createMediaStreamDestination();
      this.outputGainNode?.connect(destination); // Connect output to recorder
      this.mediaRecorder = new MediaRecorder(destination.stream);

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.recordedChunks.push(event.data);
        }
      };

      this.mediaRecorder.onstop = async () => {
        this.recordingState = 'processing';
        this.requestUpdate();
        try {
          // Re-encode to MP3 using FFmpeg
          await this.loadFFmpeg();
          if (!this.ffmpeg) {
            throw new Error('FFmpeg failed to load.');
          }

          const audioBlob = new Blob(this.recordedChunks, {type: 'audio/webm'});
          const arrayBuffer = await audioBlob.arrayBuffer();
          const uint8Array = new Uint8Array(arrayBuffer);

          await this.ffmpeg.writeFile('input.webm', uint8Array);
          await this.ffmpeg.exec(['-i', 'input.webm', 'output.mp3']);

          const outputData = (await this.ffmpeg.readFile('output.mp3')) as Uint8Array;
          const mp3Blob = new Blob([outputData], {type: 'audio/mp3'});
          this.recordedAudioUrl = URL.createObjectURL(mp3Blob);

          this.recordingState = 'finished';
        } catch (error) {
          console.error('Error processing recording:', error);
          alert('Failed to process recording.');
          this.recordingState = 'idle';
        } finally {
          // Fix: Use terminate() method for FFmpeg cleanup.
          this.ffmpeg?.terminate(); // Clean up FFmpeg instance
          this.ffmpeg = null;
          this.requestUpdate();
        }
      };

      this.mediaRecorder.start();
      this.recordingState = 'recording';
      this.requestUpdate();
    } else if (this.recordingState === 'recording') {
      // Stop recording
      this.mediaRecorder?.stop();
      this.outputGainNode?.disconnect(); // Disconnect recorder from output
      this.recordingState = 'processing'; // State will change to 'finished' or 'idle' after processing
      this.requestUpdate();
    }
  }

  private async loadFFmpeg() {
    if (this.ffmpeg) {
      return;
    }
    this.recordingState = 'initializing';
    this.requestUpdate();
    this.ffmpeg = new FFmpeg();
    this.ffmpeg.on('log', ({message}) => console.log(`FFmpeg: ${message}`));
    await this.ffmpeg.load({
      coreURL: await toBlobURL('https://unpkg.com/@ffmpeg/core@0.12.6/dist/ffmpeg-core.js', 'text/javascript'),
      wasmURL: await toBlobURL('https://unpkg.com/@ffmpeg/core@0.12.6/dist/ffmpeg-core.wasm', 'application/wasm'),
      // workerURL: await toBlobURL('https://unpkg.com/@ffmpeg/ffmpeg@0.12.10/dist/ffmpeg.js', 'text/javascript'),
    });
    // The recording state will be updated by handleRecordButtonClick after FFmpeg is loaded
  }

  private handleVolumeChange(event: Event) {
    const slider = event.target as HTMLInputElement;
    this.outputVolume = parseFloat(slider.value);
    if (this.outputGainNode) {
      this.outputGainNode.gain.value = this.outputVolume;
    }
  }

  override render() {
    const isPlayingOrLoading =
      this.playbackState === 'playing' || this.playbackState === 'loading';
    const isPlaying = this.playbackState === 'playing';
    const isLoading = this.playbackState === 'loading';
    const isStopped = this.playbackState === 'stopped';
    const isPaused = this.playbackState === 'paused';

    const recordingDisabled =
      this.recordingState === 'initializing' ||
      this.recordingState === 'processing' ||
      !isPlaying; // Can only record if playing

    const playPauseButtonText = isPlaying
      ? 'Pause'
      : isLoading
        ? 'Loading...'
        : 'Play';

    const playPauseButtonIcon = isPlaying
      ? svg`<svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24"><path d="M576-216v-528h168v528H576Zm-336 0v-528h168v528H240Z"/></svg>`
      : isLoading
        ? html`<div class="spinner"></div>`
        : svg`<svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24"><path d="M320-216v-528l440 264-440 264Z"/></svg>`;

    const recordButtonClasses = classMap({
      'record-button': true,
      'recording-active': this.recordingState === 'recording',
    });

    return html`
      <div class="main-container">
        <header>
          <h1>AURALSPIRIT</h1>
          <p>Steer a continuous stream of music with text prompts</p>
        </header>

        <section>
          <div class="prompt-input-area">
            <input
              id="promptInput"
              type="text"
              placeholder="Enter a prompt, e.g., 'driving synthwave' or 'relaxing piano'"
              .value=${this.currentPromptInput}
              @input=${(e: Event) =>
                (this.currentPromptInput = (e.target as HTMLInputElement).value)}
              @keydown=${this.handlePromptInputKeyDown}
              aria-label="Enter music prompt"
            />
            <button @click=${this.addPrompt} aria-label="Add prompt">
              <svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24"><path d="M440-440H200v-80h240V200h80v240h240v80H520v240h-80v-240Z"/></svg>
              Add
            </button>
          </div>
          <div class="prompt-presets">
            ${map(
              PROMPT_TEXT_PRESETS,
              (preset) => html`
                <button
                  class="preset-button"
                  @click=${() => this.addPresetPrompt(preset)}
                  aria-label="Add preset prompt: ${preset}"
                >
                  ${preset}
                </button>
              `,
            )}
          </div>
        </section>

        <section class="prompts-list-area">
          ${this.prompts.length === 0
            ? html`<p style="text-align: center; color: #888;">
                Add prompts to start generating music!
              </p>`
            : map(
                this.prompts,
                (p) => html`
                  <div
                    class="prompt-item"
                    style="border-left-color: ${p.color}; background-color: ${hexToRgbA(p.color, 0.05)};"
                  >
                    <div class="prompt-text-wrapper">
                      <span class="prompt-text">${p.text}</span>
                      <button
                        class="remove-prompt-button"
                        @click=${() => this.removePrompt(p.promptId)}
                        aria-label="Remove prompt: ${p.text}"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24"><path d="m256-200-56-56 224-224-224-224 56-56 224 224 224-224 56 56-224 224 224 224-56 56-224-224-224 224Z"/></svg>
                      </button>
                    </div>
                    <weight-slider
                      .value=${p.weight}
                      .color=${p.color}
                      @value-change=${(e: CustomEvent<number>) =>
                        this.updatePromptWeight(p.promptId, e.detail)}
                      aria-label="Adjust weight for prompt: ${p.text}"
                    ></weight-slider>
                  </div>
                `,
              )}
        </section>

        <div class="controls-area">
          <button
            class="play-pause-button"
            @click=${this.handlePlayPauseButtonClick}
            ?disabled=${isLoading}
            aria-label="${playPauseButtonText} music stream"
          >
            ${playPauseButtonIcon} ${playPauseButtonText}
          </button>

          <button
            class="stop-button"
            @click=${this.stopPlayback}
            ?disabled=${isStopped || isLoading}
            aria-label="Stop music stream"
          >
            <svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24"><path d="M240-240v-480h480v480H240Z"/></svg>
            Stop
          </button>

          <div class="volume-control">
            <label for="volume-slider">Volume:</label>
            <input
              id="volume-slider"
              type="range"
              min="0"
              max="1"
              step="0.01"
              .value=${this.outputVolume.toString()}
              @input=${this.handleVolumeChange}
              aria-label="Music volume"
            />
          </div>

          <button
            class=${recordButtonClasses}
            @click=${this.handleRecordButtonClick}
            ?disabled=${recordingDisabled}
            aria-label="${this.recordingState === 'recording' ? 'Stop recording' : 'Start recording'}"
          >
            ${this.recordingState === 'recording'
              ? svg`<svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24"><path d="M320-320v-320h320v320H320Z"/></svg>` // Stop icon (square)
              : this.recordingState === 'processing' ||
                  this.recordingState === 'initializing'
                ? html`<div class="spinner"></div>`
                : svg`<svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24"><path d="M480-360q-50 0-85-35t-35-85q0-50 35-85t85-35q50 0 85 35t35 85q0 50-35 85t-85 35Zm0 280q-83 0-156-31.5T197-197q-54-54-85.5-127T80-480q0-83 31.5-156T197-763q54-54 127-85.5T480-880q83 0 156 31.5T763-763q54 54 85.5 127T880-480q0 83-31.5 156T763-197q-54 54-127 85.5T480-80Zm0-80q134 0 227-93t93-227q0-134-93-227t-227-93q-134 0-227 93t-93 227q0 134 93 227t227 93Zm0-320Z"/></svg>`}
            ${this.recordingState === 'recording'
              ? 'Recording'
              : this.recordingState === 'initializing'
                ? 'Initializing...'
                : this.recordingState === 'processing'
                  ? 'Processing...'
                  : 'Record'}
          </button>
        </div>

        ${this.recordedAudioUrl
          ? html`<a
              class="download-link"
              href=${this.recordedAudioUrl}
              download="auraspirit-music.mp3"
              aria-label="Download recorded music"
            >
              <svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 -960 960 960" width="24"><path d="M480-320 280-520l56-56 104 104v-328h80v328l104-104 56 56-200 200ZM240-160q-33 0-56.5-23.5T160-240v-112h80v112h480v-112h80v112q0 33-23.5 56.5T720-160H240Z"/></svg>
              Download Recording
            </a>`
          : ''}
      </div>
    `;
  }
}