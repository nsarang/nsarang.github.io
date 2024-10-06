// config.js
export const CONFIG = {
    API_BASE_URL: 'https://huggingface.co/api',
    MODEL_WEIGHT_EXTENSIONS: ['.bin', '.pt', '.safetensors'],
    DEFAULT_MODEL: 'Llama-3.2-1B-Instruct-q4f32_1-MLC',
    DEFAULT_TEMPERATURE: 0.8,
    DEFAULT_MAX_TOKENS: 1000,
    DEFAULT_TOP_P: 0.9,
    DEFAULT_FREQUENCY_PENALTY: 0,
    DEFAULT_PRESENCE_PENALTY: 0,
  };

// utils.js
export const debounce = (func, delay) => {
    let timeoutId;
    return (...args) => {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => func(...args), delay);
    };
};

export const formatBytes = (bytes, decimals = 2) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
};

// api.js

export const getFileSizeFromURL = async (fileUrl) => {
    try {
        const response = await fetch(fileUrl, { method: 'HEAD' });
        const contentLength = response.headers.get('Content-Length');
        return contentLength ? parseInt(contentLength, 10) : 0;
    } catch (error) {
        console.error(`Error fetching size for ${fileUrl}:`, error);
        return 0;
    }
};

export const estimateRepoSize = async (repoId) => {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/models/${repoId}`);
        const repoData = await response.json();

        if (!repoData.siblings) {
            throw new Error('No file metadata found in the repository');
        }

        const modelFiles = repoData.siblings.filter(file =>
            CONFIG.MODEL_WEIGHT_EXTENSIONS.some(ext => file.rfilename.endsWith(ext))
        );

        const totalSizeBytes = await modelFiles.reduce(async (accPromise, file) => {
            const acc = await accPromise;
            if (typeof file.size === 'number') {
                return acc + file.size;
            } else if (file.lfs && file.lfs.size) {
                return acc + file.lfs.size;
            } else {
                const fileUrl = `https://huggingface.co/${repoId}/resolve/main/${file.rfilename}`;
                const fileSize = await getFileSizeFromURL(fileUrl);
                return acc + fileSize;
            }
        }, Promise.resolve(0));

        return (totalSizeBytes / (1024 * 1024)).toFixed(1);
    } catch (error) {
        console.error('Error fetching repository data:', error);
        throw error;
    }
};

// ui.js
import { marked } from 'https://cdn.jsdelivr.net/npm/marked/lib/marked.esm.js';
import hljs from 'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.10.0/build/es/highlight.min.js';

// Initialize highlight.js
hljs.highlightAll();

// Configure marked to use highlight.js for code syntax highlighting
marked.setOptions({
  highlight: function(code, lang) {
    const language = hljs.getLanguage(lang) ? lang : 'plaintext';
    return hljs.highlight(code, { language }).value;
  },
  langPrefix: 'hljs language-'
});

export const updateUI = {
    modelInfo: (size) => {
        const infoElement = document.getElementById('model-info');
        infoElement.innerHTML = size === 'loading'
            ? '<span class="loading-icon"></span> Estimating model size...'
            : `Estimated model size: ${size}`;
    },
    downloadStatus: (text) => {
        const statusElement = document.getElementById('download-status');
        statusElement.textContent = text;
        statusElement.classList.remove('hidden');
    },
    initializeButton: (text, disabled) => {
        const button = document.getElementById('initialize-model');
        button.textContent = text;
        button.disabled = disabled;
    },
    sendButton: (disabled) => {
        document.getElementById('send').disabled = disabled;
    },
    appendMessage: (message) => {
        const chatBox = document.getElementById('chat-messages');
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${message.role}-message`);
        
        // Render markdown for assistant messages
        if (message.role === 'assistant') {
          messageElement.innerHTML = marked.parse(message.content);
        } else {
          messageElement.textContent = message.content;
        }
        
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    
        // Apply syntax highlighting to code blocks
        messageElement.querySelectorAll('pre code').forEach((block) => {
          hljs.highlightElement(block);
        });
    },
    updateLastMessage: (content) => {
        const chatBox = document.getElementById('chat-messages');
        const lastMessage = chatBox.lastElementChild;
        if (lastMessage) {
          // Render markdown for the updated content
          lastMessage.innerHTML = marked.parse(content);
    
          // Apply syntax highlighting to code blocks
          lastMessage.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
          });
        }
      },
    clearChat: () => {
        document.getElementById('chat-messages').innerHTML = '';
    },
    showError: (message) => {
        const errorElement = document.createElement('div');
        errorElement.classList.add('error-message');
        errorElement.textContent = message;
        document.getElementById('chat-messages').prepend(errorElement);
        setTimeout(() => errorElement.remove(), 5000);
    },
    enableParameterInputs: (enabled) => {
        const inputs = ['temperature', 'max-tokens', 'top-p', 'frequency-penalty', 'presence-penalty'];
        inputs.forEach(id => {
            document.getElementById(id).disabled = !enabled;
        });
    },
    updateProgress: (progress) => {
        const progressElement = document.getElementById('download-status');
        progressElement.textContent = `Downloading: ${(progress * 100).toFixed(2)}%`;
    },
};


// modelConfig.js
const createModelConfig = () => {
    const config = {
      temperature: CONFIG.DEFAULT_TEMPERATURE,
      max_tokens: CONFIG.DEFAULT_MAX_TOKENS,
      top_p: CONFIG.DEFAULT_TOP_P,
      frequency_penalty: CONFIG.DEFAULT_FREQUENCY_PENALTY,
      presence_penalty: CONFIG.DEFAULT_PRESENCE_PENALTY,
    };
  
    return {
      getConfig: () => ({ ...config }),
      updateConfig: (key, value) => {
        if (key in config) {
          config[key] = value;
        }
      },
    };
};

export const modelConfig = createModelConfig();

// index.js
import * as webllm from 'https://esm.run/@mlc-ai/web-llm';
// import { estimateRepoSize } from './api.js';
// import { updateUI } from './ui.js';
// import { modelConfig } from './modelConfig.js';
// import { CONFIG, debounce, formatBytes } from './utils.js';

let selectedModel = CONFIG.DEFAULT_MODEL;
let isModelInitialized = false;
const engine = new webllm.MLCEngine();
const messages = [
    { content: `
You are an advanced AI assistant, designed to be helpful and knowledgeable across a wide range of topics.
Your primary goal is to assist users with information, analysis, and creative tasks while maintaining a respectful and professional demeanor.

Guidelines:
1. Provide accurate, up-to-date information to the best of your knowledge.
2. If unsure about something, express uncertainty and suggest ways to find reliable information.
3. Maintain a neutral stance on controversial topics, presenting multiple viewpoints when appropriate.
4. Respect user privacy and avoid asking for or storing personal information.
6. Adapt your language and explanations to the user's apparent level of understanding.
7. Offer creative solutions and brainstorm ideas when asked.
8. Provide step-by-step explanations for complex topics or processes.
9. Use markdown formatting for code snippets, lists, and emphasis when appropriate.
10. Engage in follow-up questions to clarify user needs and provide more accurate assistance.

Capabilities:
- General knowledge: History, science, culture, current events (up to your knowledge cutoff date)
- Language: Translation, grammar explanations, writing assistance
- Math and logic: Basic to advanced calculations, problem-solving
- Creative writing: Storytelling, poetry, dialogue creation
- Code and technology: Programming help, explanations of tech concepts
- Analysis: Data interpretation, trend identification, critical thinking
- Task planning: Breaking down complex tasks, suggesting approaches

When responding:
1. Start with a brief, direct answer to the user's question if applicable.
2. Provide context and additional information as needed.
3. If relevant, offer examples or analogies to illustrate your points.
4. Suggest follow-up questions or areas for further exploration.
5. If asked about events after your knowledge cutoff date, remind the user of your limitations and suggest they verify the information from current sources.

Remember to approach each query with curiosity and a desire to assist the user to the best of your abilities.
        `,
      role: 'system' },
];

const updateEngineInitProgressCallback = (report) => {
    console.log("initialize", report.progress);
    updateUI.downloadStatus(report.text);
    // updateUI.updateProgress(report.progress);
};

engine.setInitProgressCallback(updateEngineInitProgressCallback);

const initializeWebLLMEngine = async () => {
    updateUI.downloadStatus('Initializing model...');
    updateUI.initializeButton('Initializing...', true);
    selectedModel = document.getElementById('model-selection').value;

    try {
        await engine.reload(selectedModel, modelConfig.getConfig());
        isModelInitialized = true;
        updateUI.sendButton(false);
        updateUI.initializeButton('Model Initialized', false);
        updateUI.enableParameterInputs(true);
    } catch (error) {
        console.error('Error initializing model:', error);
        updateUI.downloadStatus('Error initializing model. Please try again.');
        updateUI.initializeButton('Initialize Model', false);
    }
};

const streamingGenerating = async (messages, onUpdate, onFinish, onError) => {
    try {
        let curMessage = "";
        let usage;
        const config = modelConfig.getConfig();
        const completion = await engine.chat.completions.create({
            stream: true,
            messages,
            ...config,
            stream_options: { include_usage: true },
        });
        for await (const chunk of completion) {
            const curDelta = chunk.choices[0]?.delta.content;
            if (curDelta) {
                curMessage += curDelta;
            }
            if (chunk.usage) {
                usage = chunk.usage;
            }
            onUpdate(curMessage);
        }
        const finalMessage = await engine.getMessage();
        onFinish(finalMessage, usage);
    } catch (err) {
        onError(err);
    }
};

const onMessageSend = async () => {
    const input = document.getElementById('user-input').value.trim();
    if (input.length === 0) return;

    if (!isModelInitialized) {
        updateUI.showError("Please initialize the model before sending a message.");
        return;
    }

    const message = { content: input, role: 'user' };
    messages.push(message);
    updateUI.appendMessage(message);

    document.getElementById('user-input').value = '';
    updateUI.sendButton(true);

    const aiMessage = { content: '', role: 'assistant' };
    updateUI.appendMessage(aiMessage);

    const onFinishGenerating = (finalMessage, usage) => {
        updateUI.updateLastMessage(finalMessage);
        updateUI.sendButton(false);
        console.log("Usage:", usage);
    };

    streamingGenerating(
        messages,
        updateUI.updateLastMessage,
        onFinishGenerating,
        handleError
    );
};

const updateModelInfo = debounce(async () => {
    const selectedModel = document.getElementById('model-selection').value;
    updateUI.modelInfo('loading');

    try {
        const modelConfig = webllm.prebuiltAppConfig.model_list.find(m => m.model_id === selectedModel);
        if (modelConfig && modelConfig.model) {
            const repoId = modelConfig.model.replace('https://huggingface.co/', '');
            const size = await estimateRepoSize(repoId);
            updateUI.modelInfo(formatBytes(size * 1024 * 1024));
        } else {
            updateUI.modelInfo('Model information not available');
        }
    } catch (error) {
        updateUI.modelInfo('Unable to estimate model size');
    }
}, 300);

const handleError = (error) => {
    console.error("Error:", error);
    updateUI.showError("An error occurred. Please try again.");
    updateUI.sendButton(false);
};

const syncUIWithConfig = () => {
    const config = modelConfig.getConfig();
    
    document.getElementById('model-selection').value = selectedModel;
    document.getElementById('temperature').value = config.temperature;
    document.getElementById('temperature-value').textContent = config.temperature;
    document.getElementById('max-tokens').value = config.max_tokens;
    document.getElementById('top-p').value = config.top_p;
    document.getElementById('top-p-value').textContent = config.top_p;
    document.getElementById('frequency-penalty').value = config.frequency_penalty;
    document.getElementById('frequency-penalty-value').textContent = config.frequency_penalty;
    document.getElementById('presence-penalty').value = config.presence_penalty;
    document.getElementById('presence-penalty-value').textContent = config.presence_penalty;
  };

const setupEventListeners = () => {
    document.getElementById('initialize-model').addEventListener('click', initializeWebLLMEngine);
    document.getElementById('send').addEventListener('click', onMessageSend);
    document.getElementById('clear-chat').addEventListener('click', () => {
        updateUI.clearChat();
        messages.length = 1; // Keep only the system message
    });

    document.getElementById('user-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            onMessageSend();
        }
    });

    document.getElementById('model-selection').addEventListener('change', updateModelInfo);

    ['temperature', 'max-tokens', 'top-p', 'frequency-penalty', 'presence-penalty'].forEach(id => {
        const element = document.getElementById(id);
        element.addEventListener('input', (e) => {
          const value = parseFloat(e.target.value);
          document.getElementById(`${id}-value`).textContent = value;
          modelConfig.updateConfig(id.replace('-', '_'), value);
        });
      });
};

const initializeApp = () => {
    const modelSelect = document.getElementById('model-selection');
    webllm.prebuiltAppConfig.model_list.forEach((m) => {
        const option = document.createElement('option');
        option.value = m.model_id;
        option.textContent = m.model_id;
        modelSelect.appendChild(option);
    });
    modelSelect.value = selectedModel;
    
    syncUIWithConfig();
    updateModelInfo();
    setupEventListeners();
    updateUI.enableParameterInputs(true);
};

document.addEventListener('DOMContentLoaded', initializeApp);