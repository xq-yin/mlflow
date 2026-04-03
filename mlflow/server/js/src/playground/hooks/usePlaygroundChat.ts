import { useState, useRef, useCallback } from 'react';
import { getDefaultHeaders } from '../../common/utils/FetchUtils';

export interface PlaygroundMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  isError?: boolean;
}

export interface PlaygroundChatConfig {
  requestUrl: string;
  systemPrompt: string;
  temperature: number;
  maxTokens: number | undefined;
  endpointName: string;
}

interface UsePlaygroundChatReturn {
  messages: PlaygroundMessage[];
  isStreaming: boolean;
  error: string | null;
  sendMessage: (content: string, config: PlaygroundChatConfig) => void;
  stopStreaming: () => void;
  clearChat: () => void;
}

let messageIdCounter = 0;
const generateId = () => `msg-${Date.now()}-${++messageIdCounter}`;

/**
 * Parses SSE lines from a ReadableStream chunk buffer.
 * Yields the data payload of each `data: ...` line.
 */
function extractSSEData(buffer: string): { dataChunks: string[]; remaining: string } {
  const lines = buffer.split('\n');
  const remaining = lines.pop() || '';
  const dataChunks: string[] = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('data: ')) {
      const data = trimmed.slice(6);
      if (data !== '[DONE]') {
        dataChunks.push(data);
      }
    }
  }

  return { dataChunks, remaining };
}

export function usePlaygroundChat(): UsePlaygroundChatReturn {
  const [messages, setMessages] = useState<PlaygroundMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const streamingContentRef = useRef('');

  const sendMessage = useCallback(
    async (content: string, config: PlaygroundChatConfig) => {
      const { requestUrl, systemPrompt, temperature, maxTokens, endpointName } = config;

      if (!requestUrl || !content.trim()) return;

      setError(null);

      const userMessage: PlaygroundMessage = {
        id: generateId(),
        role: 'user',
        content: content.trim(),
        timestamp: new Date(),
      };

      const assistantMessage: PlaygroundMessage = {
        id: generateId(),
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMessage, assistantMessage]);
      setIsStreaming(true);
      streamingContentRef.current = '';

      // Build the messages array for the API
      const apiMessages: { role: string; content: string }[] = [];
      if (systemPrompt.trim()) {
        apiMessages.push({ role: 'system', content: systemPrompt.trim() });
      }
      // Include all previous messages (excluding the new ones we just added)
      for (const msg of messages) {
        if (msg.role !== 'system') {
          apiMessages.push({ role: msg.role, content: msg.content });
        }
      }
      apiMessages.push({ role: 'user', content: content.trim() });

      const payload: Record<string, unknown> = {
        messages: apiMessages,
        stream: true,
      };
      if (temperature !== undefined) payload['temperature'] = temperature;
      if (maxTokens !== undefined) payload['max_tokens'] = maxTokens;
      // For chat-completions endpoint, include model name
      if (requestUrl.includes('/v1/chat/completions')) {
        payload['model'] = endpointName;
      }

      const abortController = new AbortController();
      abortControllerRef.current = abortController;

      try {
        const response = await fetch(requestUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...getDefaultHeaders(document.cookie),
          },
          body: JSON.stringify(payload),
          signal: abortController.signal,
        });

        if (!response.ok) {
          let errorText: string;
          try {
            const errorJson = await response.json();
            errorText = errorJson.detail || errorJson.message || errorJson.error || response.statusText;
          } catch {
            errorText = await response.text().catch(() => response.statusText);
          }
          throw new Error(`${response.status}: ${errorText}`);
        }

        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error('Response body is not readable');
        }

        const decoder = new TextDecoder();
        let buffer = '';

        let readerDone = false;
        while (!readerDone) {
          const { done, value } = await reader.read();
          if (done) {
            readerDone = true;
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          const { dataChunks, remaining } = extractSSEData(buffer);
          buffer = remaining;

          for (const chunk of dataChunks) {
            try {
              const parsed = JSON.parse(chunk);
              const delta = parsed.choices?.[0]?.delta?.content;
              if (delta) {
                streamingContentRef.current += delta;
                const updatedContent = streamingContentRef.current;
                setMessages((prev) => {
                  const updated = [...prev];
                  const last = updated[updated.length - 1];
                  if (last && last.role === 'assistant') {
                    updated[updated.length - 1] = { ...last, content: updatedContent };
                  }
                  return updated;
                });
              }
            } catch {
              // Skip unparseable chunks
            }
          }
        }

        // Finalize the assistant message
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last && last.role === 'assistant') {
            updated[updated.length - 1] = { ...last, isStreaming: false };
          }
          return updated;
        });
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          // User cancelled, finalize message as-is
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last && last.role === 'assistant') {
              updated[updated.length - 1] = { ...last, isStreaming: false };
            }
            return updated;
          });
        } else {
          const errorMessage = err instanceof Error ? err.message : String(err);
          setError(errorMessage);
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last && last.role === 'assistant') {
              updated[updated.length - 1] = {
                ...last,
                content: errorMessage,
                isStreaming: false,
                isError: true,
              };
            }
            return updated;
          });
        }
      } finally {
        setIsStreaming(false);
        abortControllerRef.current = null;
      }
    },
    [messages],
  );

  const stopStreaming = useCallback(() => {
    abortControllerRef.current?.abort();
  }, []);

  const clearChat = useCallback(() => {
    abortControllerRef.current?.abort();
    setMessages([]);
    setError(null);
    setIsStreaming(false);
    streamingContentRef.current = '';
  }, []);

  return { messages, isStreaming, error, sendMessage, stopStreaming, clearChat };
}
