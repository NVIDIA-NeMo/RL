/**
 * AI Assistant Main Coordinator
 * Brings together all AI Assistant modules and provides unified interface
 */

// Prevent multiple initializations
if (typeof window.AIAssistant !== 'undefined') {
    // Already loaded, skip
} else {

    class AIAssistant {
        constructor(options = {}) {
            this.options = {
                enableAI: options.enableAI !== false, // Default to enabled
                assistantApiKey: options.assistantApiKey || '',
                assistantEndpoint: options.assistantEndpoint || '',
                aiTriggerThreshold: options.aiTriggerThreshold || 2, // Trigger AI if fewer than N results
                autoTrigger: options.autoTrigger !== false, // Default to auto-trigger
                debounceDelay: options.debounceDelay || 2000, // 2 seconds for RAG
                ...options
            };

            this.isLoaded = false;

            // Module instances
            this.aiClient = null;
            this.responseProcessor = null;
            this.responseRenderer = null;
            this.markdownProcessor = null;

            this.init();
        }

        async init() {
            try {
                // Load required modules
                await this.loadModules();

                // Initialize modules
                this.markdownProcessor = new MarkdownProcessor();
                this.responseProcessor = new ResponseProcessor();
                this.responseRenderer = new ResponseRenderer(this.markdownProcessor);
                this.aiClient = new AIClient(this.options);

                this.isLoaded = true;
            } catch (error) {
                console.error('❌ Failed to initialize AI Assistant:', error);
                this.fallbackToBasicMode();
            }
        }

        async loadModules() {
            const moduleNames = [
                'MarkdownProcessor',
                'ResponseRenderer',
                'ResponseProcessor',
                'AIClient'
            ];

            // Load modules with smart path resolution
            const modulePromises = moduleNames.map(name =>
                this.loadModuleWithFallback(name)
            );

            await Promise.all(modulePromises);
        }

        async loadModuleWithFallback(moduleName) {
            const possiblePaths = this.getModulePaths(moduleName);

            for (const path of possiblePaths) {
                try {
                    await this.loadModule(path);
                    return;
                } catch (error) {
                    // Try next path
                }
            }

            throw new Error(`Failed to load AI Assistant module ${moduleName} from any path`);
        }

        getModulePaths(moduleName) {
            const fileName = `${moduleName}.js`;

            // Calculate nesting level to determine correct _static path
            const pathParts = window.location.pathname.split('/').filter(part => part.length > 0);
            const htmlFile = pathParts[pathParts.length - 1];

            // Remove the HTML file from the count if it exists
            let nestingLevel = pathParts.length;
            if (htmlFile && htmlFile.endsWith('.html')) {
                nestingLevel--;
            }

            // Build the correct _static path based on nesting level
            const staticPrefix = nestingLevel > 0 ? '../'.repeat(nestingLevel) : './';
            const staticPath = `${staticPrefix}_static`;

            // Determine the correct subdirectory for each module
            let moduleDir = '';
            if (['AIClient', 'ResponseProcessor'].includes(moduleName)) {
                moduleDir = 'core';
            } else if (['MarkdownProcessor', 'ResponseRenderer'].includes(moduleName)) {
                moduleDir = 'ui';
            }

            // Generate paths in order of likelihood
            const paths = [];

            // 1. Most likely path based on calculated nesting
            if (moduleDir) {
                paths.push(`${staticPath}/${moduleDir}/${fileName}`);
            }

            // 2. Fallback static paths (for different nesting scenarios)
            if (moduleDir) {
                paths.push(`_static/${moduleDir}/${fileName}`);
                paths.push(`./_static/${moduleDir}/${fileName}`);
                if (nestingLevel > 1) {
                    paths.push(`../_static/${moduleDir}/${fileName}`);
                }
            }

            return paths;
        }

        async loadModule(src) {
            // Check if module is already loaded to prevent duplicates
            const scriptId = `ai-module-${src.split('/').pop().replace('.js', '')}`;
            if (document.getElementById(scriptId)) {
                return Promise.resolve();
            }

            return new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = src;
                script.id = scriptId;
                script.onload = resolve;
                script.onerror = () => reject(new Error(`Failed to load module: ${src}`));
                document.head.appendChild(script);
            });
        }

        // Public API methods
        async analyzeQuery(query, searchResults = []) {
            if (!this.aiClient) {
                console.warn('🤖 AI Client not yet initialized');
                return null;
            }

            try {
                // Get response from AI client
                const rawResponse = await this.aiClient.analyzeQuery(query, searchResults);

                if (!rawResponse) {
                    return null;
                }

                // Process response through response processor
                const processedResponse = this.responseProcessor.hasCachedResponse(query)
                    ? this.responseProcessor.getCachedResponse(query)
                    : this.responseProcessor.processResponse(query, rawResponse);

                return processedResponse;
            } catch (error) {
                console.error('🤖 Error in analyzeQuery:', error);
                return this.responseProcessor.processError(error, query);
            }
        }

        renderResponse(aiResponse, query) {
            if (!this.responseRenderer) {
                console.warn('🤖 Response Renderer not yet initialized');
                return '<div class="alert alert-warning">AI Assistant not ready</div>';
            }

            return this.responseRenderer.renderResponse(aiResponse, query);
        }

        renderError(message) {
            if (!this.responseRenderer) {
                return `<div class="alert alert-danger">AI Assistant Error: ${message}</div>`;
            }

            return this.responseRenderer.renderError(message);
        }

        renderLoading() {
            if (!this.responseRenderer) {
                return '<div class="alert alert-info">Loading AI Assistant...</div>';
            }

            return this.responseRenderer.renderLoading();
        }

        renderManualTrigger(onTrigger) {
            if (!this.responseRenderer) {
                return '<div class="alert alert-info">AI Assistant not ready</div>';
            }

            return this.responseRenderer.renderManualTrigger(onTrigger);
        }

        // Convenience methods
        markdownToHtml(markdown) {
            if (!this.markdownProcessor) {
                console.warn('🤖 Markdown Processor not yet initialized');
                return markdown;
            }

            return this.markdownProcessor.markdownToHtml(markdown);
        }

        clearCache() {
            if (this.responseProcessor) {
                this.responseProcessor.clearCache();
            }
        }

        getCacheSize() {
            return this.responseProcessor ? this.responseProcessor.getCacheSize() : 0;
        }

        isAvailable() {
            return this.aiClient ? this.aiClient.isAvailable() : false;
        }

        isLoading() {
            return this.aiClient ? this.aiClient.isLoading() : false;
        }

        // Fallback mode
        fallbackToBasicMode() {
            // Provide basic functionality without modules
            this.isLoaded = false;
        }

        // Get module instances (for advanced usage)
        getAIClient() {
            return this.aiClient;
        }

        getResponseProcessor() {
            return this.responseProcessor;
        }

        getResponseRenderer() {
            return this.responseRenderer;
        }

        getMarkdownProcessor() {
            return this.markdownProcessor;
        }

        getOptions() {
            return this.options;
        }
    }

    // Make AIAssistant available globally
    window.AIAssistant = AIAssistant;

    // Auto-initialize if not already done
    document.addEventListener('DOMContentLoaded', function () {
        if (!window.aiAssistantInstance) {
            // Create the global instance
            window.aiAssistantInstance = new AIAssistant();

            // Emit initialization event
            const initEvent = new CustomEvent('ai-assistant-initialized', {
                detail: { instance: window.aiAssistantInstance }
            });
            document.dispatchEvent(initEvent);
        }
    });

} // End of if check for preventing multiple loads

console.log('🤖 AI Assistant main coordinator loaded successfully'); 