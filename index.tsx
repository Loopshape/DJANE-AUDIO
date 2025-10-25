/**
 * @fileoverview Control real time music with text prompts
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import {css, CSSResultGroup, html, LitElement, svg} from 'lit';
import {customElement, property, query, state}from 'lit/decorators.js';
import {classMap} from 'lit/directives/class-map.js';
import {styleMap} from 'lit/directives/style-map.js';

import {
  GoogleGenAI,
  type LiveMusicGenerationConfig,
  type LiveMusicServerMessage,
  type LiveMusicSession,
  Modality,
} from '@google/genai';
import {decode, decodeAudioData} from './utils';
import {FFmpeg} from '@ffmpeg/ffmpeg';
import {fetchFile, toBlobURL} from '@ffmpeg/util';

const ai = new GoogleGenAI({
  apiKey: process.env.API_KEY,
  apiVersion: 'v1alpha',
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

  constructor() {
    super();
    this.handlePointerDown = this.handlePointerDown.bind(this);
    this.handlePointerMove = this.handlePointerMove.bind(this);
    this.handleTouchMove = this.handleTouchMove.bind(this);
    this.handlePointerUp = this.handlePointerUp.bind(this);
  }

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
    this.containerBounds = null;
  }

  private handleWheel(e: WheelEvent) {
    e.preventDefault();
    const delta = e.deltaY;
    this.value = this.value + delta * -0.005;
    this.value = Math.max(0, Math.min(2, this.value));
    this.dispatchInputEvent();
  }

  private updateValueFromPosition(clientY: number) {
    if (!this.containerBounds) return;

    const trackHeight = this.containerBounds.height;
    // Calculate position relative to the top of the track
    const relativeY = clientY - this.containerBounds.top;
    // Invert and normalize (0 at bottom, 1 at top)
    const normalizedValue =
      1 - Math.max(0, Math.min(trackHeight, relativeY)) / trackHeight;
    // Scale to 0-2 range
    this.value = normalizedValue * 2;

    this.dispatchInputEvent();
  }

  private dispatchInputEvent() {
    this.dispatchEvent(new CustomEvent<number>('input', {detail: this.value}));
  }

  override render() {
    const thumbHeightPercent = (this.value / 2) * 100;
    const thumbStyle = styleMap({
      height: `${thumbHeightPercent}%`,
      backgroundColor: this.color,
      // Hide thumb if value is 0 or very close to prevent visual glitch
      display: this.value > 0.01 ? 'block' : 'none',
    });
    const displayValue = this.value.toFixed(2);

    return html`
      <div
        class="scroll-container"
        @pointerdown=${this.handlePointerDown}
        @wheel=${this.handleWheel}>
        <div class="slider-container">
          <div id="thumb" style=${thumbStyle}></div>
        </div>
        <div class="value-display">${displayValue}</div>
      </div>
    `;
  }
}

// Base class for icon buttons.
class IconButton extends LitElement {
  static override styles = css`
    :host {
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
      pointer-events: none;
    }
    :host(:hover) svg {
      transform: scale(1.2);
    }
    svg {
      width: 100%;
      height: 100%;
      transition: transform 0.5s cubic-bezier(0.25, 1.56, 0.32, 0.99);
    }
    .hitbox {
      pointer-events: all;
      position: absolute;
      width: 65%;
      aspect-ratio: 1;
      top: 9%;
      border-radius: 50%;
      cursor: pointer;
    }
  ` as CSSResultGroup;

  // Method to be implemented by subclasses to provide the specific icon SVG
  protected renderIcon() {
    return svg``; // Default empty icon
  }

  private renderSVG() {
    return html` <svg
      width="140"
      height="140"
      viewBox="0 -10 140 150"
      fill="none"
      xmlns="http://www.w3.org/2000/svg">
      <rect
        x="22"
        y="6"
        width="96"
        height="96"
        rx="48"
        fill="black"
        fill-opacity="0.05" />
      <rect
        x="23.5"
        y="7.5"
        width="93"
        height="93"
        rx="46.5"
        stroke="black"
        stroke-opacity="0.3"
        stroke-width="3" />
      <g filter="url(#filter0_ddi_1048_7373)">
        <rect
          x="25"
          y="9"
          width="90"
          height="90"
          rx="45"
          fill="white"
          fill-opacity="0.05"
          shape-rendering="crispEdges" />
      </g>
      ${this.renderIcon()}
      <defs>
        <filter
          id="filter0_ddi_1048_7373"
          x="0"
          y="0"
          width="140"
          height="140"
          filterUnits="userSpaceOnUse"
          color-interpolation-filters="sRGB">
          <feFlood flood-opacity="0" result="BackgroundImageFix" />
          <feColorMatrix
            in="SourceAlpha"
            type="matrix"
            values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0"
            result="hardAlpha" />
          <feOffset dy="2" />
          <feGaussianBlur stdDeviation="4" />
          <feComposite in2="hardAlpha" operator="out" />
          <feColorMatrix
            type="matrix"
            values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.25 0" />
          <feBlend
            mode="normal"
            in2="BackgroundImageFix"
            result="effect1_dropShadow_1048_7373" />
          <feColorMatrix
            in="SourceAlpha"
            type="matrix"
            values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0"
            result="hardAlpha" />
          <feOffset dy="16" />
          <feGaussianBlur stdDeviation="12.5" />
          <feComposite in2="hardAlpha" operator="out" />
          <feColorMatrix
            type="matrix"
            values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.25 0" />
          <feBlend
            mode="normal"
            in2="effect1_dropShadow_1048_7373"
            result="effect2_dropShadow_1048_7373" />
          <feBlend
            mode="normal"
            in="SourceGraphic"
            in2="effect2_dropShadow_1048_7373"
            result="shape" />
          <feColorMatrix
            in="SourceAlpha"
            type="matrix"
            values="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 127 0"
            result="hardAlpha" />
          <feOffset dy="3" />
          <feGaussianBlur stdDeviation="1.5" />
          <feComposite in2="hardAlpha" operator="arithmetic" k2="-1" k3="1" />
          <feColorMatrix
            type="matrix"
            values="0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0.05 0" />
          <feBlend
            mode="normal"
            in2="shape"
            result="effect3_innerShadow_1048_7373" />
        </filter>
      </defs>
    </svg>`;
  }

  override render() {
    return html`${this.renderSVG()}<div class="hitbox"></div>`;
  }
}

// PlayPauseButton
// -----------------------------------------------------------------------------

/** A button for toggling play/pause. */
@customElement('play-pause-button')
export class PlayPauseButton extends IconButton {
  @property({type: String}) playbackState: PlaybackState = 'stopped';

  static override styles = [
    IconButton.styles,
    css`
      .loader {
        stroke: #ffffff;
        stroke-width: 3;
        stroke-linecap: round;
        animation: spin linear 1s infinite;
        transform-origin: center;
        transform-box: fill-box;
      }
      @keyframes spin {
        from {
          transform: rotate(0deg);
        }
        to {
          transform: rotate(359deg);
        }
      }
    `,
  ];

  private renderPause() {
    return svg`<path
      d="M75.0037 69V39H83.7537V69H75.0037ZM56.2537 69V39H65.0037V69H56.2537Z"
      fill="#FEFEFE"
    />`;
  }

  private renderPlay() {
    return svg`<path d="M60 71.5V36.5L87.5 54L60 71.5Z" fill="#FEFEFE" />`;
  }

  private renderLoading() {
    return svg`<path shape-rendering="crispEdges" class="loader" d="M70,74.2L70,74.2c-10.7,0-19.5-8.7-19.5-19.5l0,0c0-10.7,8.7-19.5,19.5-19.5
            l0,0c10.7,0,19.5,8.7,19.5,19.5l0,0"/>`;
  }

  override renderIcon() {
    if (this.playbackState === 'playing') {
      return this.renderPause();
    } else if (this.playbackState === 'loading') {
      return this.renderLoading();
    } else {
      return this.renderPlay();
    }
  }
}

@customElement('reset-button')
export class ResetButton extends IconButton {
  private renderResetIcon() {
    return svg`<path fill="#fefefe" d="M71,77.1c-2.9,0-5.7-0.6-8.3-1.7s-4.8-2.6-6.7-4.5c-1.9-1.9-3.4-4.1-4.5-6.7c-1.1-2.6-1.7-5.3-1.7-8.3h4.7
      c0,4.6,1.6,8.5,4.8,11.7s7.1,4.8,11.7,4.8c4.6,0,8.5-1.6,11.7-4.8c3.2-3.2,4.8-7.1,4.8-11.7s-1.6-8.5-4.8-11.7
      c-3.2-3.2-7.1-4.8-11.7-4.8h-0.4l3.7,3.7L71,46.4L61.5,37l9.4-9.4l3.3,3.4l-3.7,3.7H71c2.9,0,5.7,0.6,8.3,1.7
      c2.6,1.1,4.8,2.6,6.7,4.5c1.9,1.9,3.4,4.1,4.5,6.7c1.1,2.6,1.7,5.3,1.7,8.3c0,2.9-0.6,5.7-1.7,8.3c-1.1,2.6-2.6,4.8-4.5,6.7
      s-4.1,3.4-6.7,4.5C76.7,76.5,73.9,77.1,71,77.1z"/>`;
  }

  override renderIcon() {
    return this.renderResetIcon();
  }
}

// AddPromptButton component
// -----------------------------------------------------------------------------
/** A button for adding a new prompt. */
@customElement('add-prompt-button')
export class AddPromptButton extends IconButton {
  private renderAddIcon() {
    return svg`<path d="M67 40 H73 V52 H85 V58 H73 V70 H67 V58 H55 V52 H67 Z" fill="#FEFEFE" />`;
  }

  override renderIcon() {
    return this.renderAddIcon();
  }
}

@customElement('record-button')
export class RecordButton extends IconButton {
  @property({type: String}) recordingState: RecordingState = 'idle';

  static override styles = [
    IconButton.styles,
    css`
      .loader {
        stroke: #ffffff;
        stroke-width: 3;
        stroke-linecap: round;
        animation: spin linear 1s infinite;
        transform-origin: center;
        transform-box: fill-box;
      }
      @keyframes spin {
        from {
          transform: rotate(0deg);
        }
        to {
          transform: rotate(359deg);
        }
      }
    `,
  ];

  private renderRecordIcon() {
    return svg`<circle cx="70" cy="54" r="18" fill="#FF4136"/>`;
  }

  private renderStopIcon() {
    return svg`<rect x="52" y="36" width="36" height="36" fill="#FEFEFE"/>`;
  }

  private renderProcessingIcon() {
    return svg`<path shape-rendering="crispEdges" class="loader" d="M70,74.2L70,74.2c-10.7,0-19.5-8.7-19.5-19.5l0,0c0-10.7,8.7-19.5,19.5-19.5
            l0,0c10.7,0,19.5,8.7,19.5,19.5l0,0"/>`;
  }

  override renderIcon() {
    switch (this.recordingState) {
      case 'recording':
        return this.renderStopIcon();
      case 'processing':
      case 'initializing':
        return this.renderProcessingIcon();
      case 'idle':
      case 'finished':
      default:
        return this.renderRecordIcon();
    }
  }
}

@customElement('download-button')
export class DownloadButton extends IconButton {
  private renderDownloadIcon() {
    return svg`<path fill="#fefefe" d="M70,34.3l-28.2,28.2l6.6,6.6L65,52.5v40.4h10V52.5l16.6,16.6l6.6-6.6L70,34.3z M40 90 H100 V100 H40z"/>`;
  }

  override renderIcon() {
    return this.renderDownloadIcon();
  }
}

// Toast Message component
// -----------------------------------------------------------------------------

@customElement('toast-message')
class ToastMessage extends LitElement {
  static override styles = css`
    .toast {
      line-height: 1.6;
      position: fixed;
      top: 20px;
      left: 50%;
      transform: translateX(-50%);
      background-color: #000;
      color: white;
      padding: 15px;
      border-radius: 5px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 15px;
      min-width: 200px;
      max-width: 80vw;
      transition: transform 0.5s cubic-bezier(0.19, 1, 0.22, 1);
      z-index: 11;
    }
    button {
      border-radius: 100px;
      aspect-ratio: 1;
      border: none;
      color: #000;
      cursor: pointer;
    }
    .toast:not(.showing) {
      transition-duration: 1s;
      transform: translate(-50%, -200%);
    }
  `;

  @property({type: String}) message = '';
  @property({type: Boolean}) showing = false;

  override render() {
    return html`<div class=${classMap({showing: this.showing, toast: true})}>
      <div class="message">${this.message}</div>
      <button @click=${this.hide}>✕</button>
    </div>`;
  }

  show(message: string) {
    this.showing = true;
    this.message = message;
  }

  hide() {
    this.showing = false;
  }
}

/** A single prompt input */
@customElement('prompt-controller')
class PromptController extends LitElement {
  static override styles = css`
    .prompt {
      position: relative;
      height: 100%;
      width: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      box-sizing: border-box;
      overflow: hidden;
      background-color: #2a2a2a;
      border-radius: 5px;
      transition:
        opacity 0.3s,
        border-color 0.3s;
      border: 2px solid transparent;
    }
    .remove-button {
      position: absolute;
      top: 1.2vmin;
      left: 1.2vmin;
      background: #666;
      color: #fff;
      border: none;
      border-radius: 50%;
      width: 2.8vmin;
      height: 2.8vmin;
      font-size: 1.8vmin;
      display: flex;
      align-items: center;
      justify-content: center;
      line-height: 2.8vmin;
      cursor: pointer;
      opacity: 0.5;
      transition: opacity 0.2s;
      z-index: 10;
    }
    .remove-button:hover {
      opacity: 1;
    }
    weight-slider {
      /* Calculate height: 100% of parent minus controls height and margin */
      max-height: calc(100% - 9vmin);
      flex: 1;
      min-height: 10vmin;
      width: 100%;
      box-sizing: border-box;
      overflow: hidden;
      margin: 2vmin 0 1vmin;
    }
    .controls {
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
      align-items: center;
      gap: 0.2vmin;
      width: 100%;
      height: 8vmin;
      padding: 0 0.5vmin;
      box-sizing: border-box;
      margin-bottom: 1vmin;
    }
    #text {
      font-family: 'Google Sans', sans-serif;
      font-size: 1.8vmin;
      width: 100%;
      flex-grow: 1;
      max-height: 100%;
      padding: 0.4vmin;
      box-sizing: border-box;
      text-align: center;
      word-wrap: break-word;
      overflow-y: auto;
      border: none;
      outline: none;
      -webkit-font-smoothing: antialiased;
      color: #fff;
      scrollbar-width: thin;
      scrollbar-color: #666 #1a1a1a;
      background: transparent;
    }
    #text::-webkit-scrollbar {
      width: 6px;
    }
    #text::-webkit-scrollbar-track {
      background: #0009;
      border-radius: 3px;
    }
    #text::-webkit-scrollbar-thumb {
      background-color: #666;
      border-radius: 3px;
    }
    :host([filtered]) .prompt {
      opacity: 0.5;
      border-color: #da2000;
    }
  `;

  @property({type: Boolean, reflect: true}) filtered = false;
  @property({type: String, reflect: true}) promptId = '';
  @property({type: String}) text = '';
  @property({type: Number}) weight = 0;
  @property({type: String}) color = '';

  @query('weight-slider') private weightInput!: WeightSlider;
  @query('#text') private textInput!: HTMLSpanElement;

  private handleTextKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter') {
      e.preventDefault();
      this.updateText();
      (e.target as HTMLElement).blur();
    }
  }

  private dispatchPromptChange() {
    this.dispatchEvent(
      new CustomEvent<Prompt>('prompt-changed', {
        detail: {
          promptId: this.promptId,
          text: this.text,
          weight: this.weight,
          color: this.color,
        },
      }),
    );
  }

  private updateText() {
    console.log('updateText');
    const newText = this.textInput.textContent?.trim();
    if (newText === '') {
      this.textInput.textContent = this.text;
      return;
    }
    this.text = newText!;
    this.dispatchPromptChange();
  }

  private updateWeight() {
    this.weight = this.weightInput.value;
    this.dispatchPromptChange();
  }

  private dispatchPromptRemoved() {
    this.dispatchEvent(
      new CustomEvent<string>('prompt-removed', {
        detail: this.promptId,
        bubbles: true,
        composed: true,
      }),
    );
  }

  override render() {
    const classes = classMap({
      'prompt': true,
    });
    return html`<div class=${classes}>
      <button class="remove-button" @click=${this.dispatchPromptRemoved}
        >×</button
      >
      <weight-slider
        id="weight"
        value=${this.weight}
        color=${this.color}
        @input=${this.updateWeight}></weight-slider>
      <div class="controls">
        <span
          id="text"
          spellcheck="false"
          contenteditable="plaintext-only"
          @keydown=${this.handleTextKeyDown}
          @blur=${this.updateText}
          >${this.text}</span
        >
      </div>
    </div>`;
  }
}

/** A dedicated component for vocal track input. */
@customElement('vocal-channel')
class VocalChannel extends LitElement {
  static override styles = css`
    :host {
      height: 100%;
      max-height: 80vmin;
      min-width: 24vmin;
      max-width: 30vmin;
      flex: 1;
      display: flex;
      flex-direction: column;
      box-sizing: border-box;
      overflow: hidden;
      background-color: #2a2a2a;
      border-radius: 5px;
      padding: 1.5vmin;
      font-family: 'Google Sans', sans-serif;
    }
    .header {
      font-weight: bold;
      font-size: 1.8vmin;
      color: #fff;
      margin-bottom: 1vmin;
      width: 100%;
      text-align: left;
    }
    textarea {
      flex-grow: 1;
      resize: none;
      background-color: #1f1f1f;
      color: #eee;
      border: 1px solid #666;
      border-radius: 3px;
      padding: 0.8vmin;
      font-size: 1.5vmin;
      font-family: inherit;
      box-sizing: border-box;
      scrollbar-width: thin;
      scrollbar-color: #666 #1a1a1a;
      margin-bottom: 1vmin;
    }
    textarea::-webkit-scrollbar {
      width: 6px;
    }
    textarea::-webkit-scrollbar-track {
      background: #1a1a1a;
      border-radius: 3px;
    }
    textarea::-webkit-scrollbar-thumb {
      background-color: #666;
      border-radius: 3px;
    }
    textarea:focus {
      outline: none;
      border-color: #ff25f6;
      box-shadow: 0 0 0 2px rgba(255, 37, 246, 0.3);
    }
    .play-button {
      background-color: #ff25f6;
      color: white;
      border: none;
      border-radius: 5px;
      padding: 1vmin;
      font-size: 1.6vmin;
      cursor: pointer;
      font-family: inherit;
      transition: background-color 0.2s;
      flex-shrink: 0;
      height: 5vmin;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .play-button:hover:not(:disabled) {
      background-color: #e022d8;
    }
    .play-button:disabled {
      background-color: #666;
      cursor: not-allowed;
      opacity: 0.7;
    }
  `;

  @property({type: String}) lyrics = '';
  @property({type: String}) status: 'idle' | 'loading' | 'playing' = 'idle';

  private handleLyricsInput(e: Event) {
    this.lyrics = (e.target as HTMLTextAreaElement).value;
  }

  private handlePlayClick() {
    if (this.status !== 'idle' || !this.lyrics.trim()) return;
    this.dispatchEvent(
      new CustomEvent('play-vocals', {
        detail: this.lyrics,
        bubbles: true,
        composed: true,
      }),
    );
  }

  override render() {
    let buttonText = 'Play Vocals';
    if (this.status === 'loading') {
      buttonText = 'Generating...';
    } else if (this.status === 'playing') {
      buttonText = 'Playing...';
    }
    const disabled = this.status !== 'idle' || !this.lyrics.trim();

    return html`
      <div class="header">Vocal Sampler</div>
      <textarea
        id="lyrics"
        .value=${this.lyrics}
        @input=${this.handleLyricsInput}
        placeholder="Enter lyrics for TTS..."></textarea>
      <button
        class="play-button"
        @click=${this.handlePlayClick}
        .disabled=${disabled}>
        ${buttonText}
      </button>
    `;
  }
}

/** A panel for managing real-time music generation settings. */
@customElement('settings-controller')
class SettingsController extends LitElement {
  static override styles = css`
    :host {
      display: block;
      padding: 2vmin;
      background-color: #2a2a2a;
      color: #eee;
      box-sizing: border-box;
      border-radius: 5px;
      font-family: 'Google Sans', sans-serif;
      font-size: 1.5vmin;
      overflow-y: auto;
      scrollbar-width: thin;
      scrollbar-color: #666 #1a1a1a;
      transition: width 0.3s ease-out max-height 0.3s ease-out;
    }
    :host([showadvanced]) {
      max-height: 40vmin;
    }
    :host::-webkit-scrollbar {
      width: 6px;
    }
    :host::-webkit-scrollbar-track {
      background: #1a1a1a;
      border-radius: 3px;
    }
    :host::-webkit-scrollbar-thumb {
      background-color: #666;
      border-radius: 3px;
    }
    .setting {
      margin-bottom: 0.5vmin;
      display: flex;
      flex-direction: column;
      gap: 0.5vmin;
    }
    label {
      font-weight: bold;
      display: flex;
      justify-content: space-between;
      align-items: center;
      white-space: nowrap;
      user-select: none;
    }
    label span:last-child {
      font-weight: normal;
      color: #ccc;
      min-width: 3em;
      text-align: right;
    }
    input[type='range'] {
      --track-height: 8px;
      --track-bg: #0009;
      --track-border-radius: 4px;
      --thumb-size: 16px;
      --thumb-bg: #5200ff;
      --thumb-border-radius: 50%;
      --thumb-box-shadow: 0 0 3px rgba(0, 0, 0, 0.7);
      --value-percent: 0%;
      -webkit-appearance: none;
      appearance: none;
      width: 100%;
      height: var(--track-height);
      background: transparent;
      cursor: pointer;
      margin: 0.5vmin 0;
      border: none;
      padding: 0;
      vertical-align: middle;
    }
    input[type='range']::-webkit-slider-runnable-track {
      width: 100%;
      height: var(--track-height);
      cursor: pointer;
      border: none;
      background: linear-gradient(
        to right,
        var(--thumb-bg) var(--value-percent),
        var(--track-bg) var(--value-percent)
      );
      border-radius: var(--track-border-radius);
    }
    input[type='range']::-moz-range-track {
      width: 100%;
      height: var(--track-height);
      cursor: pointer;
      background: var(--track-bg);
      border-radius: var(--track-border-radius);
      border: none;
    }
    input[type='range']::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      height: var(--thumb-size);
      width: var(--thumb-size);
      background: var(--thumb-bg);
      border-radius: var(--thumb-border-radius);
      box-shadow: var(--thumb-box-shadow);
      cursor: pointer;
      margin-top: calc((var(--thumb-size) - var(--track-height)) / -2);
    }
    input[type='range']::-moz-range-thumb {
      height: var(--thumb-size);
      width: var(--thumb-size);
      background: var(--thumb-bg);
      border-radius: var(--thumb-border-radius);
      box-shadow: var(--thumb-box-shadow);
      cursor: pointer;
      border: none;
    }
    input[type='number'],
    input[type='text'],
    select,
    textarea {
      background-color: #2a2a2a;
      color: #eee;
      border: 1px solid #666;
      border-radius: 3px;
      padding: 0.4vmin;
      font-size: 1.5vmin;
      font-family: inherit;
      box-sizing: border-box;
    }
    input[type='number'] {
      width: 6em;
    }
    input[type='text'] {
      width: 100%;
    }
    textarea {
      width: 100%;
      min-height: 8vmin;
      resize: vertical;
    }
    input[type='text']::placeholder,
    textarea::placeholder {
      color: #888;
    }
    input[type='number']:focus,
    input[type='text']:focus,
    textarea:focus {
      outline: none;
      border-color: #5200ff;
      box-shadow: 0 0 0 2px rgba(82, 0, 255, 0.3);
    }
    select {
      width: 100%;
    }
    select:focus {
      outline: none;
      border-color: #5200ff;
    }
    select option {
      background-color: #2a2a2a;
      color: #eee;
    }
    .checkbox-setting {
      flex-direction: row;
      align-items: center;
      gap: 1vmin;
    }
    .core-settings-row {
      display: flex;
      flex-direction: row;
      flex-wrap: wrap;
      gap: 4vmin;
      margin-bottom: 1vmin;
      justify-content: space-evenly;
    }
    .core-settings-row .setting {
      min-width: 16vmin;
    }
    .core-settings-row label span:last-child {
      min-width: 2.5em;
    }
    .advanced-toggle {
      cursor: pointer;
      margin: 2vmin 0 1vmin 0;
      color: #aaa;
      text-decoration: underline;
      user-select: none;
      font-size: 1.4vmin;
      width: fit-content;
    }
    .advanced-toggle:hover {
      color: #eee;
    }
    .advanced-settings {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(10vmin, 1fr));
      gap: 3vmin;
      overflow: hidden;
      max-height: 0;
      opacity: 0;
      transition:
        max-height 0.3s ease-out,
        opacity 0.3s ease-out;
    }
    .advanced-settings.visible {
      max-width: 120vmin;
      max-height: 40vmin;
      opacity: 1;
    }
    hr.divider {
      display: none;
      border: none;
      border-top: 1px solid #666;
      margin: 2vmin 0;
      width: 100%;
    }
    :host([showadvanced]) hr.divider {
      display: block;
    }
    .auto-row {
      display: flex;
      align-items: center;
      gap: 0.5vmin;
    }
    .setting[auto='true'] input[type='range'] {
      pointer-events: none;
      filter: grayscale(100%);
    }
    .auto-row span {
      margin-left: auto;
    }
    .auto-row label {
      cursor: pointer;
    }
    .auto-row input[type='checkbox'] {
      cursor: pointer;
      margin: 0;
    }
  `;

  // Fix: Per the error, MusicGenerationMode.INSTRUMENTAL does not exist.
  // In this version of the API, instrumental is the default mode when
  // `musicGenerationMode` is not specified.
  private readonly defaultConfig: LiveMusicGenerationConfig = {
    temperature: 1.1,
    topK: 40,
    guidance: 4.0,
  };

  @property({attribute: false})
  config: LiveMusicGenerationConfig = this.defaultConfig;

  @state() showAdvanced = false;
  @state() autoDensity = true;
  @state() lastDefinedDensity: number | undefined;
  @state() autoBrightness = true;
  @state() lastDefinedBrightness: number | undefined;

  public resetToDefaults() {
    this.config = this.defaultConfig;
    this.autoDensity = true;
    this.lastDefinedDensity = undefined;
    this.autoBrightness = true;
    this.lastDefinedBrightness = undefined;
    this.dispatchSettingsChange();
  }

  private updateSliderBackground(inputEl: HTMLInputElement) {
    if (inputEl.type !== 'range') {
      return;
    }
    const min = Number(inputEl.min) || 0;
    const max = Number(inputEl.max) || 100;
    const value = Number(inputEl.value);
    const percentage = ((value - min) / (max - min)) * 100;
    inputEl.style.setProperty('--value-percent', `${percentage}%`);
  }

  private handleInputChange(e: Event) {
    const target = e.target as
      | HTMLInputElement
      | HTMLSelectElement
      | HTMLTextAreaElement;
    const key = target.id;
    let value: string | number | boolean | undefined = target.value;

    if (target.type === 'number' || target.type === 'range') {
      value = target.value === '' ? undefined : Number(target.value);
      if (target.type === 'range') {
        this.updateSliderBackground(target as HTMLInputElement);
      }
    } else if (target.type === 'checkbox') {
      value = (target as HTMLInputElement).checked;
    } else if (target.type === 'select-one') {
      const selectElement = target as HTMLSelectElement;
      if (selectElement.options[selectElement.selectedIndex]?.disabled) {
        value = undefined;
      }
    }

    const newConfig: LiveMusicGenerationConfig & {[key: string]: any} = {
      ...this.config,
    };
    newConfig[key] = value;

    if (newConfig.density !== undefined) {
      this.lastDefinedDensity = newConfig.density;
    }
    if (newConfig.brightness !== undefined) {
      this.lastDefinedBrightness = newConfig.brightness;
    }
    if (key === 'auto-density') {
      this.autoDensity = Boolean(value);
      newConfig.density = this.autoDensity
        ? undefined
        : this.lastDefinedDensity;
    } else if (key === 'auto-brightness') {
      this.autoBrightness = Boolean(value);
      newConfig.brightness = this.autoBrightness
        ? undefined
        : this.lastDefinedBrightness;
    }

    this.config = newConfig;
    this.dispatchSettingsChange();
  }

  override updated(changedProperties: Map<string | symbol, unknown>) {
    super.updated(changedProperties);
    if (changedProperties.has('config')) {
      this.shadowRoot
        ?.querySelectorAll<HTMLInputElement>('input[type="range"]')
        .forEach((slider: HTMLInputElement) => {
          const configValue =
            this.config[slider.id as keyof LiveMusicGenerationConfig];
          if (typeof configValue === 'number') {
            slider.value = String(configValue);
          } else if (slider.id === 'density' || slider.id === 'brightness') {
            slider.value = String(configValue ?? 0.5);
          }
          this.updateSliderBackground(slider);
        });
    }
  }

  private dispatchSettingsChange() {
    this.dispatchEvent(
      new CustomEvent<LiveMusicGenerationConfig>('settings-changed', {
        detail: this.config,
        bubbles: true,
        composed: true,
      }),
    );
  }

  private toggleAdvancedSettings() {
    this.showAdvanced = !this.showAdvanced;
  }

  override render() {
    const cfg = this.config;
    const advancedClasses = classMap({
      'advanced-settings': true,
      'visible': this.showAdvanced,
    });
    const scaleMap = new Map<string, string>([
      ['Auto', 'SCALE_UNSPECIFIED'],
      ['C Major / A Minor', 'C_MAJOR_A_MINOR'],
      ['C# Major / A# Minor', 'D_FLAT_MAJOR_B_FLAT_MINOR'],
      ['D Major / B Minor', 'D_MAJOR_B_MINOR'],
      ['D# Major / C Minor', 'E_FLAT_MAJOR_C_MINOR'],
      ['E Major / C# Minor', 'E_MAJOR_D_FLAT_MINOR'],
      ['F Major / D Minor', 'F_MAJOR_D_MINOR'],
      ['F# Major / D# Minor', 'G_FLAT_MAJOR_E_FLAT_MINOR'],
      ['G Major / E Minor', 'G_MAJOR_E_MINOR'],
      ['G# Major / F Minor', 'A_FLAT_MAJOR_F_MINOR'],
      ['A Major / F# Minor', 'A_MAJOR_G_FLAT_MINOR'],
      ['A# Major / G Minor', 'B_FLAT_MAJOR_G_MINOR'],
      ['B Major / G# Minor', 'B_MAJOR_A_FLAT_MINOR'],
    ]);

    return html`
      <div class="core-settings-row">
        <div class="setting">
          <label for="temperature"
            >Temperature<span>${cfg.temperature!.toFixed(1)}</span></label
          >
          <input
            type="range"
            id="temperature"
            min="0"
            max="3"
            step="0.1"
            .value=${cfg.temperature!.toString()}
            @input=${this.handleInputChange} />
        </div>
        <div class="setting">
          <label for="guidance"
            >Guidance<span>${cfg.guidance!.toFixed(1)}</span></label
          >
          <input
            type="range"
            id="guidance"
            min="0"
            max="6"
            step="0.1"
            .value=${cfg.guidance!.toString()}
            @input=${this.handleInputChange} />
        </div>
        <div class="setting">
          <label for="topK">Top K<span>${cfg.topK}</span></label>
          <input
            type="range"
            id="topK"
            min="1"
            max="100"
            step="1"
            .value=${cfg.topK!.toString()}
            @input=${this.handleInputChange} />
        </div>
      </div>
      <hr class="divider" />
      <div class=${advancedClasses}>
        <div class="setting">
          <label for="seed">Seed</label>
          <input
            type="number"
            id="seed"
            .value=${cfg.seed ?? ''}
            @input=${this.handleInputChange}
            placeholder="Auto" />
        </div>
        <div class="setting">
          <label for="bpm">BPM</label>
          <input
            type="number"
            id="bpm"
            min="60"
            max="180"
            .value=${cfg.bpm ?? ''}
            @input=${this.handleInputChange}
            placeholder="Auto" />
        </div>
        <div class="setting" auto=${this.autoDensity}>
          <label for="density">Density</label>
          <input
            type="range"
            id="density"
            min="0"
            max="1"
            step="0.05"
            .value=${(this.lastDefinedDensity ?? 0.5).toString()}
            @input=${this.handleInputChange} />
          <div class="auto-row">
            <input
              type="checkbox"
              id="auto-density"
              .checked=${this.autoDensity}
              @input=${this.handleInputChange} />
            <label for="auto-density">Auto</label>
            <span>${(this.lastDefinedDensity ?? 0.5).toFixed(2)}</span>
          </div>
        </div>
        <div class="setting" auto=${this.autoBrightness}>
          <label for="brightness">Brightness</label>
          <input
            type="range"
            id="brightness"
            min="0"
            max="1"
            step="0.05"
            .value=${(this.lastDefinedBrightness ?? 0.5).toString()}
            @input=${this.handleInputChange} />
          <div class="auto-row">
            <input
              type="checkbox"
              id="auto-brightness"
              .checked=${this.autoBrightness}
              @input=${this.handleInputChange} />
            <label for="auto-brightness">Auto</label>
            <span>${(this.lastDefinedBrightness ?? 0.5).toFixed(2)}</span>
          </div>
        </div>
        <div class="setting">
          <label for="scale">Scale</label>
          <select
            id="scale"
            .value=${cfg.scale || 'SCALE_UNSPECIFIED'}
            @change=${this.handleInputChange}>
            <option value="" disabled selected>Select Scale</option>
            ${[...scaleMap.entries()].map(
              ([displayName, enumValue]) =>
                html`<option value=${enumValue}>${displayName}</option>`,
            )}
          </select>
        </div>
        <div class="setting">
          <div class="setting checkbox-setting">
            <input
              type="checkbox"
              id="muteBass"
              .checked=${!!cfg.muteBass}
              @change=${this.handleInputChange} />
            <label for="muteBass" style="font-weight: normal;">Mute Bass</label>
          </div>
          <div class="setting checkbox-setting">
            <input
              type="checkbox"
              id="muteDrums"
              .checked=${!!cfg.muteDrums}
              @change=${this.handleInputChange} />
            <label for="muteDrums" style="font-weight: normal;"
              >Mute Drums</label
            >
          </div>
          <div class="setting checkbox-setting">
            <input
              type="checkbox"
              id="onlyBassAndDrums"
              .checked=${!!cfg.onlyBassAndDrums}
              @change=${this.handleInputChange} />
            <label for="onlyBassAndDrums" style="font-weight: normal;"
              >Only Bass & Drums</label
            >
          </div>
        </div>
      </div>
      <div class="advanced-toggle" @click=${this.toggleAdvancedSettings}>
        ${this.showAdvanced ? 'Hide' : 'Show'} Advanced Settings
      </div>
    `;
  }
}

/** Component for the PromptDJ UI. */
@customElement('prompt-dj')
class PromptDj extends LitElement {
  static override styles = css`
    :host {
      height: 100%;
      width: 100%;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      box-sizing: border-box;
      padding: 2vmin;
      position: relative;
      font-size: 1.8vmin;
    }
    :host([locked='add-reset']) reset-button,
    :host([locked='add-reset']) .add-prompt-button-container {
      pointer-events: none;
      opacity: 0.6;
      transition: opacity 0.3s;
    }
    #background {
      position: absolute;
      height: 100%;
      width: 100%;
      z-index: -1;
      background: #111;
    }
    .prompts-area {
      display: flex;
      align-items: flex-end;
      justify-content: center;
      flex: 4;
      width: 100%;
      margin-top: 2vmin;
      gap: 2vmin;
    }
    #prompts-container {
      display: flex;
      flex-direction: row;
      align-items: flex-end;
      flex-shrink: 1;
      height: 100%;
      gap: 2vmin;
      padding: 1vmin;
      overflow-x: auto;
      scrollbar-width: thin;
      scrollbar-color: #666 #1a1a1a;
    }
    #prompts-container::-webkit-scrollbar {
      height: 8px;
    }
    #prompts-container::-webkit-scrollbar-track {
      background: #111;
      border-radius: 4px;
    }
    #prompts-container::-webkit-scrollbar-thumb {
      background-color: #666;
      border-radius: 4px;
    }
    #prompts-container::-webkit-scrollbar-thumb:hover {
      background-color: #777;
    }
    /* Add pseudo-elements for centering while keeping elements visible when scrolling */
    #prompts-container::before,
    #prompts-container::after {
      content: '';
      flex: 1;
      min-width: 0.5vmin;
    }
    .add-prompt-button-container {
      display: flex;
      align-items: flex-end;
      height: 100%;
      flex-shrink: 0;
    }
    #settings-container {
      flex: 1;
      margin: 2vmin 0 1vmin 0;
    }
    .playback-container {
      display: flex;
      justify-content: center;
      align-items: center;
      flex-shrink: 0;
    }
    play-pause-button,
    add-prompt-button,
    reset-button,
    record-button,
    download-button {
      width: 12vmin;
      flex-shrink: 0;
    }
    prompt-controller,
    vocal-channel {
      height: 100%;
      max-height: 80vmin;
    }
    prompt-controller {
      min-width: 14vmin;
      max-width: 16vmin;
      flex: 1;
    }
    vocal-channel {
      flex-shrink: 0;
    }
  `;

  @property({
    type: Object,
    attribute: false,
  })
  private prompts: Map<string, Prompt>;
  private nextPromptId: number; // Monotonically increasing ID for new prompts
  private session!: LiveMusicSession;
  private readonly sampleRate = 48000;
  private audioContext = new (window.AudioContext ||
    (window as any).webkitAudioContext)({sampleRate: this.sampleRate});
  private outputNode: GainNode = this.audioContext.createGain();
  private nextStartTime = 0;
  private readonly bufferTime = 2; // adds an audio buffer in case of netowrk latency
  @state() private playbackState: PlaybackState = 'stopped';
  @property({type: Object})
  private filteredPrompts = new Set<string>();
  private connectionError = true;

  @state() private setupComplete = false;
  @state() private vocalStatus: 'idle' | 'loading' | 'playing' = 'idle';

  private setupPromise: Promise<void> | null = null;
  private resolveSetupPromise: (() => void) | null = null;

  @state() private recordingState: RecordingState = 'idle';
  private ffmpeg: FFmpeg | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private recordStreamDest: MediaStreamAudioDestinationNode | null = null;
  private recordedChunks: Blob[] = [];
  private recordedUrl: string | null = null;

  @query('play-pause-button') private playPauseButton!: PlayPauseButton;
  @query('toast-message') private toastMessage!: ToastMessage;
  @query('settings-controller') private settingsController!: SettingsController;
  @query('vocal-channel') private vocalChannelEl!: VocalChannel;

  private readonly throttledUpdateServer = throttle(
    this.updateServerNow.bind(this),
    300,
  );

  constructor(prompts: Map<string, Prompt>) {
    super();
    this.prompts = prompts;
    this.nextPromptId = this.prompts.size;
    this.outputNode.connect(this.audioContext.destination);
  }

  override async firstUpdated() {
    await this.connectToSession();
  }

  override updated(changedProperties: Map<string | symbol, unknown>) {
    super.updated(changedProperties);
    if (changedProperties.has('playbackState')) {
      const isLocked =
        this.playbackState === 'playing' || this.playbackState === 'loading';
      if (isLocked) {
        // Only lock Add and Reset buttons
        this.setAttribute('locked', 'add-reset');
      } else {
        this.removeAttribute('locked');
      }
    }
  }

  private async connectToSession() {
    this.setupComplete = false;
    this.setupPromise = new Promise((resolve) => {
      this.resolveSetupPromise = resolve;
    });
    this.session = await ai.live.music.connect({
      model: model,
      callbacks: {
        onmessage: async (e: LiveMusicServerMessage) => {
          console.log('Received message from the server: %s\n');
          console.log(e);
          if (e.setupComplete) {
            this.connectionError = false;
            this.setupComplete = true;
            if (this.resolveSetupPromise) {
              this.resolveSetupPromise();
              this.resolveSetupPromise = null;
            }
          }
          if (e.filteredPrompt) {
            this.filteredPrompts = new Set([
              ...this.filteredPrompts,
              e.filteredPrompt.text,
            ]);
            this.toastMessage.show(
              `Prompt filtered: ${e.filteredPrompt.filteredReason}`,
            );
          }
          if (e.serverContent?.audioChunks !== undefined) {
            if (
              this.playbackState === 'paused' ||
              this.playbackState === 'stopped'
            )
              return;
            const audioBuffer = await decodeAudioData(
              decode(e.serverContent?.audioChunks[0].data),
              this.audioContext,
              48000,
              2,
            );
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.outputNode);
            if (this.nextStartTime === 0) {
              this.nextStartTime =
                this.audioContext.currentTime + this.bufferTime;
              setTimeout(() => {
                this.playbackState = 'playing';
              }, this.bufferTime * 1000);
            }

            if (this.nextStartTime < this.audioContext.currentTime) {
              console.log('under run');
              this.playbackState = 'loading';
              this.nextStartTime = 0;
              return;
            }
            source.start(this.nextStartTime);
            this.nextStartTime += audioBuffer.duration;
          }
        },
        onerror: (e: ErrorEvent) => {
          console.error('Session error:', e);
          this.connectionError = true;
          this.stopAudio();
          this.toastMessage.show(
            'A connection error occurred. Please try playing again.',
          );
        },
        onclose: (e: CloseEvent) => {
          console.log(`Connection closed: ${e.code} ${e.reason}`);
          this.connectionError = true;
          this.stopAudio();
          let message = 'Connection closed. Please try playing again.';
          if (e.reason) {
            message = `Connection closed: ${e.reason}. Please try again.`;
          }
          this.toastMessage.show(message);
        },
      },
    });
  }

  private async updateServerNow() {
    if (!this.session || !this.setupComplete) {
      console.warn(
        'updateServerNow called before session is ready. Ignoring.',
      );
      return;
    }
    try {
      // Logic simplified: always instrumental mode.
      const musicConfig: any = {...this.settingsController.config};

      const promptsToSend = Array.from(this.prompts.values()).filter((p) => {
        return !this.filteredPrompts.has(p.text) && p.weight !== 0;
      });
      const weightedPrompts = promptsToSend.map((p) => ({
        text: p.text,
        weight: p.weight,
      }));

      // If playing, we need at least one prompt for instrumental music.
      if (this.playbackState !== 'stopped' && weightedPrompts.length === 0) {
        // Don't send an update that would cause an error. The music will
        // continue playing with the last valid set of prompts.
        return;
      }

      await this.session.setMusicGenerationConfig({
        musicGenerationConfig: musicConfig,
      });
      await this.session.setWeightedPrompts({weightedPrompts});
    } catch (e: any) {
      console.error('Failed to update settings:', e);
      this.toastMessage.show(`API Error (updating settings): ${e.message}`);
      this.pauseAudio();
    }
  }

  private dispatchPromptsChange() {
    this.dispatchEvent(
      new CustomEvent('prompts-changed', {detail: this.prompts}),
    );
  }

  private handlePromptChanged(e: CustomEvent<Prompt>) {
    const {promptId, text, weight} = e.detail;
    const prompt = this.prompts.get(promptId);

    if (!prompt) {
      console.error('prompt not found', promptId);
      return;
    }

    prompt.text = text;
    prompt.weight = weight;

    const newPrompts = new Map(this.prompts);
    newPrompts.set(promptId, prompt);

    this.prompts = newPrompts;
    this.requestUpdate();
    this.dispatchPromptsChange();
    if (this.playbackState === 'playing' || this.playbackState === 'loading') {
      this.throttledUpdateServer();
    }
  }

  /** Generates radial gradients for each prompt based on weight and color. */
  private makeBackground() {
    const clamp01 = (v: number) => Math.min(Math.max(v, 0), 1);

    const MAX_WEIGHT = 0.5;
    const MAX_ALPHA = 0.6;

    const bg: string[] = [];

    [...this.prompts.values()].forEach((p, i) => {
      const alphaPct = clamp01(p.weight / MAX_WEIGHT) * MAX_ALPHA;
      const alpha = Math.round(alphaPct * 0xff)
        .toString(16)
        .padStart(2, '0');

      const stop = p.weight / 2;
      const x = (i % 4) / 3;
      const y = Math.floor(i / 4) / 3;
      const s = `radial-gradient(circle at ${x * 100}% ${
        y * 100
      }%, ${p.color}${alpha} 0px, ${p.color}00 ${stop * 100}%)`;

      bg.push(s);
    });

    return bg.join(', ');
  }

  private async handlePlayPause() {
    if (this.playbackState === 'playing') {
      this.pauseAudio();
    } else if (
      this.playbackState === 'paused' ||
      this.playbackState === 'stopped'
    ) {
      if (this.connectionError) {
        await this.connectToSession();
      }
      await this.loadAudio();
    } else if (this.playbackState === 'loading') {
      this.stopAudio();
    }
    console.debug('handlePlayPause');
  }

  private pauseAudio() {
    if (this.recordingState === 'recording') {
      this.mediaRecorder?.stop();
    }
    this.session.pause();
    this.playbackState = 'paused';
    this.outputNode.gain.setValueAtTime(1, this.audioContext.currentTime);
    this.outputNode.gain.linearRampToValueAtTime(
      0,
      this.audioContext.currentTime + 0.1,
    );
    this.nextStartTime = 0;
    this.outputNode = this.audioContext.createGain();
    this.outputNode.connect(this.audioContext.destination);
  }

  private async loadAudio() {
    this.audioContext.resume();

    // Prevent API error if playing instrumentals with no active prompts.
    const hasActivePrompts = Array.from(this.prompts.values()).some(
      (p) => p.weight > 0,
    );
    if (!hasActivePrompts) {
      this.toastMessage.show(
        'Please increase the weight of at least one prompt to start.',
      );
      return; // Abort loading
    }

    if (this.setupPromise) {
      await this.setupPromise;
    }
    await this.updateServerNow();
    this.session.play();
    this.playbackState = 'loading';
    this.outputNode.gain.setValueAtTime(0, this.audioContext.currentTime);
    this.outputNode.gain.linearRampToValueAtTime(
      1,
      this.audioContext.currentTime + 0.1,
    );
  }

  private stopAudio() {
    if (this.recordingState === 'recording') {
      this.mediaRecorder?.stop();
    }
    this.session.stop();
    this.playbackState = 'stopped';
    this.outputNode.gain.setValueAtTime(0, this.audioContext.currentTime);
    this.outputNode.gain.linearRampToValueAtTime(
      1,
      this.audioContext.currentTime + 0.1,
    );
    this.nextStartTime = 0;
  }

  private async handleAddPrompt() {
    const newPromptId = `prompt-${this.nextPromptId++}`;
    const usedColors = [...this.prompts.values()].map((p) => p.color);
    const newPrompt: Prompt = {
      promptId: newPromptId,
      text: 'New Prompt', // Default text
      weight: 0,
      color: getUnusedRandomColor(usedColors),
    };
    const newPrompts = new Map(this.prompts);
    newPrompts.set(newPromptId, newPrompt);
    this.prompts = newPrompts;

    // Wait for the component to update and render the new prompt.
    // Do not dispatch the prompt change event until the user has edited the prompt text.
    await this.updateComplete;

    // Find the newly added prompt controller element
    const newPromptElement = this.renderRoot.querySelector<PromptController>(
      `prompt-controller[promptId="${newPromptId}"]`,
    );
    if (newPromptElement) {
      // Scroll the prompts container to the new prompt element
      newPromptElement.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
        inline: 'end',
      });

      // Select the new prompt text
      const textSpan =
        newPromptElement.shadowRoot?.querySelector<HTMLSpanElement>('#text');
      if (textSpan) {
        textSpan.focus();
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(textSpan);
        selection?.removeAllRanges();
        selection?.addRange(range);
      }
    }
  }

  private handlePromptRemoved(e: CustomEvent<string>) {
    e.stopPropagation();
    const promptIdToRemove = e.detail;
    if (this.prompts.has(promptIdToRemove)) {
      this.prompts.delete(promptIdToRemove);
      const newPrompts = new Map(this.prompts);
      this.prompts = newPrompts;
      this.dispatchPromptsChange();
      if (this.playbackState === 'playing' || this.playbackState === 'loading') {
        this.throttledUpdateServer();
      }
    } else {
      console.warn(
        `Attempted to remove non-existent prompt ID: ${promptIdToRemove}`,
      );
    }
  }

  // Handle scrolling X-axis the prompts container.
  private handlePromptsContainerWheel(e: WheelEvent) {
    const container = e.currentTarget as HTMLElement;
    if (e.deltaX !== 0) {
      // Prevent the default browser action (like page back/forward)
      e.preventDefault();
      container.scrollLeft += e.deltaX;
    }
  }

  private async handlePlayVocalSample(e: CustomEvent<string>) {
    this.vocalStatus = 'loading';
    const lyrics = e.detail;

    try {
      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash-preview-tts',
        contents: [
          {
            parts: [
              {text: `Say in an expressive, talk-singing voice: ${lyrics}`},
            ],
          },
        ],
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: {voiceName: 'Kore'},
            },
          },
        },
      });

      const base64Audio =
        response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;

      if (base64Audio) {
        this.audioContext.resume();
        const audioBuffer = await decodeAudioData(
          decode(base64Audio),
          this.audioContext,
          24000,
          1,
        );
        const source = this.audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.outputNode);
        source.onended = () => {
          this.vocalStatus = 'idle';
        };
        source.start();
        this.vocalStatus = 'playing';
      } else {
        throw new Error('No audio data received from API.');
      }
    } catch (err: any) {
      console.error('TTS Generation Error:', err);
      this.toastMessage.show(`Vocal Generation Error: ${err.message}`);
      this.vocalStatus = 'idle';
    }
  }

  private handleSettingsChanged() {
    if (this.playbackState === 'playing' || this.playbackState === 'loading') {
      this.throttledUpdateServer();
    }
  }

  private async handleReset() {
    if (this.connectionError) {
      await this.connectToSession();
    }
    this.pauseAudio();
    this.session.resetContext();
    if (this.vocalChannelEl) {
      this.vocalChannelEl.lyrics = '';
    }
    this.vocalStatus = 'idle';
    this.settingsController.resetToDefaults();
    setTimeout(this.loadAudio.bind(this), 100);
  }

  private async loadFFmpeg() {
    if (this.ffmpeg) return;
    this.recordingState = 'initializing';
    this.toastMessage.show('Initializing recorder...');
    try {
      this.ffmpeg = new FFmpeg();
      this.ffmpeg.on('log', ({message}) => {
        console.log('[FFMPEG]', message); // For debugging
      });
      const baseURL = 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd';
      await this.ffmpeg.load({
        coreURL: await toBlobURL(
          `${baseURL}/ffmpeg-core.js`,
          'text/javascript',
        ),
        wasmURL: await toBlobURL(
          `${baseURL}/ffmpeg-core.wasm`,
          'application/wasm',
        ),
      });
    } catch (e: any) {
      console.error('Failed to load FFmpeg', e);
      this.toastMessage.show(`Error initializing recorder: ${e.message}`);
      this.recordingState = 'idle';
      this.ffmpeg = null;
    }
  }

  private async handleRecordClick() {
    if (this.recordingState === 'recording') {
      this.mediaRecorder?.stop(); // This will trigger the 'stop' event and processing
      return;
    }

    if (
      this.playbackState !== 'playing' &&
      this.playbackState !== 'loading'
    ) {
      this.toastMessage.show('Start playback before recording.');
      return;
    }

    if (!this.ffmpeg) {
      await this.loadFFmpeg();
      // If loading failed, ffmpeg will be null
      if (!this.ffmpeg) return;
    }

    this.recordedUrl = null;
    this.recordedChunks = [];
    this.recordStreamDest = this.audioContext.createMediaStreamDestination();
    this.outputNode.connect(this.recordStreamDest);

    this.mediaRecorder = new MediaRecorder(this.recordStreamDest.stream, {
      mimeType: 'audio/webm;codecs=opus',
    });

    this.mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        this.recordedChunks.push(e.data);
      }
    };

    this.mediaRecorder.onstop = () => this.processRecording();

    this.mediaRecorder.start();
    this.recordingState = 'recording';
    this.toastMessage.show('Recording started...');
  }

  private async processRecording() {
    this.recordingState = 'processing';
    this.toastMessage.show('Processing recording, please wait...');

    const inputBlob = new Blob(this.recordedChunks, {type: 'audio/webm'});
    this.recordedChunks = []; // Clear chunks

    if (this.recordStreamDest) {
      this.outputNode.disconnect(this.recordStreamDest);
      this.recordStreamDest = null;
    }

    try {
      await this.ffmpeg!.writeFile(
        'input.webm',
        await fetchFile(inputBlob),
      );
      await this.ffmpeg!.exec(['-i', 'input.webm', 'output.mp3']);
      const data = (await this.ffmpeg!.readFile('output.mp3')) as Uint8Array;
      const outputBlob = new Blob([data.buffer], {type: 'audio/mpeg'});
      this.recordedUrl = URL.createObjectURL(outputBlob);
      this.recordingState = 'finished';
      this.toastMessage.show('Recording is ready to download.');
    } catch (e: any) {
      console.error('Error during FFMPEG processing', e);
      this.toastMessage.show(`Failed to process recording: ${e.message}`);
      this.recordingState = 'idle';
      this.recordedUrl = null;
    }
  }

  private handleDownloadClick() {
    if (!this.recordedUrl) return;
    const a = document.createElement('a');
    a.href = this.recordedUrl;
    a.download = `prompt-dj-mix-${new Date().toISOString()}.mp3`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    // URL.revokeObjectURL(this.recordedUrl); // Can cause issues if downloaded again
  }

  override render() {
    const bg = styleMap({
      backgroundImage: this.makeBackground(),
    });
    return html`<div id="background" style=${bg}></div>
      <div class="prompts-area">
        <vocal-channel
          .status=${this.vocalStatus}
          @play-vocals=${this.handlePlayVocalSample}></vocal-channel>
        <div
          id="prompts-container"
          @prompt-removed=${this.handlePromptRemoved}
          @wheel=${this.handlePromptsContainerWheel}>
          ${this.renderPrompts()}
        </div>
        <div class="add-prompt-button-container">
          <add-prompt-button @click=${this.handleAddPrompt}></add-prompt-button>
        </div>
      </div>
      <div id="settings-container">
        <settings-controller
          @settings-changed=${
            this.handleSettingsChanged
          }></settings-controller>
      </div>
      <div class="playback-container">
        <record-button
          .recordingState=${this.recordingState}
          @click=${this.handleRecordClick}></record-button>
        <play-pause-button
          @click=${this.handlePlayPause}
          .playbackState=${this.playbackState}></play-pause-button>
        <reset-button @click=${this.handleReset}></reset-button>
        ${this.recordedUrl
          ? html`<download-button
              @click=${this.handleDownloadClick}></download-button>`
          : ''}
      </div>
      <toast-message></toast-message>`;
  }

  private renderPrompts() {
    return [...this.prompts.values()].map((prompt) => {
      return html`<prompt-controller
        .promptId=${prompt.promptId}
        .filtered=${this.filteredPrompts.has(prompt.text)}
        .text=${prompt.text}
        .weight=${prompt.weight}
        .color=${prompt.color}
        @prompt-changed=${this.handlePromptChanged}>
      </prompt-controller>`;
    });
  }
}

function gen(parent: HTMLElement) {
  const initialPrompts = getStoredPrompts();

  const pdj = new PromptDj(initialPrompts);
  parent.appendChild(pdj);
}

function getStoredPrompts(): Map<string, Prompt> {
  const {localStorage} = window;
  const storedPrompts = localStorage.getItem('prompts');

  if (storedPrompts) {
    try {
      const prompts = JSON.parse(storedPrompts) as Prompt[];
      console.log('Loading stored prompts', prompts);
      return new Map(prompts.map((prompt) => [prompt.promptId, prompt]));
    } catch (e) {
      console.error('Failed to parse stored prompts', e);
    }
  }

  console.log('No stored prompts, creating prompt presets');

  const numDefaultPrompts = Math.min(4, PROMPT_TEXT_PRESETS.length);
  const shuffledPresetTexts = [...PROMPT_TEXT_PRESETS].sort(
    () => Math.random() - 0.5,
  );
  const defaultPrompts: Prompt[] = [];
  const usedColors: string[] = [];
  for (let i = 0; i < numDefaultPrompts; i++) {
    const text = shuffledPresetTexts[i];
    const color = getUnusedRandomColor(usedColors);
    usedColors.push(color);
    defaultPrompts.push({
      promptId: `prompt-${i}`,
      text,
      weight: 0,
      color,
    });
  }
  // Randomly select up to 2 prompts to set their weight to 1.
  const promptsToActivate = [...defaultPrompts].sort(() => Math.random() - 0.5);
  const numToActivate = Math.min(2, defaultPrompts.length);
  for (let i = 0; i < numToActivate; i++) {
    if (promptsToActivate[i]) {
      promptsToActivate[i].weight = 1;
    }
  }
  return new Map(defaultPrompts.map((p) => [p.promptId, p]));
}

function setStoredPrompts(prompts: Map<string, Prompt>) {
  const storedPrompts = JSON.stringify([...prompts.values()]);
  const {localStorage} = window;
  localStorage.setItem('prompts', storedPrompts);
}

function main(container: HTMLElement) {
  gen(container);
}

main(document.body);

declare global {
  interface HTMLElementTagNameMap {
    'prompt-dj': PromptDj;
    'prompt-controller': PromptController;
    'settings-controller': SettingsController;
    'add-prompt-button': AddPromptButton;
    'play-pause-button': PlayPauseButton;
    'reset-button': ResetButton;
    'weight-slider': WeightSlider;
    'toast-message': ToastMessage;
    'vocal-channel': VocalChannel;
    'record-button': RecordButton;
    'download-button': DownloadButton;
  }
}
