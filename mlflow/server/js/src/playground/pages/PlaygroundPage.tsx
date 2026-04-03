import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Button,
  CloseIcon,
  FormUI,
  GearIcon,
  Input,
  PlusIcon,
  Popover,
  RefreshIcon,
  SendIcon,
  SimpleSelect,
  SimpleSelectOption,
  SparkleDoubleIcon,
  StopIcon,
  Tag,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { usePlaygroundChat } from '../hooks/usePlaygroundChat';
import { usePlaygroundConfig } from '../hooks/usePlaygroundConfig';
import { PlaygroundMessageBubble } from '../components/PlaygroundMessageBubble';

export type PlaygroundMode = 'model-endpoint' | 'llm-judge';

const PLAYGROUND_MODE_LABELS: Record<PlaygroundMode, string> = {
  'model-endpoint': 'Model Endpoint',
  'llm-judge': 'LLM Judge',
};

const PlaygroundPage = () => {
  const { theme } = useDesignSystemTheme();
  const { config, endpoints, isLoadingEndpoints, selectEndpoint, setSystemPrompt, setTemperature, setMaxTokens } =
    usePlaygroundConfig();
  const { messages, isStreaming, error, sendMessage, stopStreaming, clearChat } = usePlaygroundChat();

  const [inputValue, setInputValue] = useState('');
  const [playgroundMode, setPlaygroundMode] = useState<PlaygroundMode>('model-endpoint');
  const [showSystemPrompt, setShowSystemPrompt] = useState(false);
  const [systemPromptDraft, setSystemPromptDraft] = useState('');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${textarea.scrollHeight}px`;
    }
  }, [inputValue]);

  const handleSend = useCallback(() => {
    if (inputValue.trim() && !isStreaming && config.requestUrl) {
      sendMessage(inputValue.trim(), {
        requestUrl: config.requestUrl,
        systemPrompt: config.systemPrompt,
        temperature: config.temperature,
        maxTokens: config.maxTokens,
        endpointName: config.endpointName,
      });
      setInputValue('');
    }
  }, [inputValue, isStreaming, config, sendMessage]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  const handleOpenSystemPrompt = useCallback(() => {
    setSystemPromptDraft(config.systemPrompt);
    setShowSystemPrompt(true);
  }, [config.systemPrompt]);

  const handleSaveSystemPrompt = useCallback(() => {
    setSystemPrompt(systemPromptDraft);
    setShowSystemPrompt(false);
  }, [systemPromptDraft, setSystemPrompt]);

  const handleCancelSystemPrompt = useCallback(() => {
    setShowSystemPrompt(false);
  }, []);

  const handleResetSystemPrompt = useCallback(() => {
    setSystemPromptDraft('');
  }, []);

  const hasEndpoint = !!config.selectedEndpoint;

  return (
    <div css={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      {/* Top bar */}
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          padding: `${theme.spacing.sm}px ${theme.spacing.lg}px`,
          borderBottom: `1px solid ${theme.colors.border}`,
          flexShrink: 0,
          gap: theme.spacing.md,
        }}
      >
        {/* Left: Title + Mode selector */}
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, flexShrink: 0 }}>
          <SparkleDoubleIcon color="ai" css={{ fontSize: 20 }} />
          <Typography.Text
            bold
            css={{
              fontSize: theme.typography.fontSizeMd + 2,
              whiteSpace: 'nowrap',
            }}
          >
            <FormattedMessage defaultMessage="Playground" description="Playground page title" />
          </Typography.Text>
          <Tag componentId="mlflow.playground.beta" color="turquoise">
            Beta
          </Tag>
        </div>

        {/* Mode selector */}
        <SimpleSelect
          id="playground-mode-select"
          componentId="mlflow.playground.mode-select"
          value={playgroundMode}
          onChange={({ target }) => setPlaygroundMode(target.value as PlaygroundMode)}
          css={{ width: 180 }}
          contentProps={{ matchTriggerWidth: true }}
        >
          {(Object.keys(PLAYGROUND_MODE_LABELS) as PlaygroundMode[]).map((mode) => (
            <SimpleSelectOption key={mode} value={mode}>
              {PLAYGROUND_MODE_LABELS[mode]}
            </SimpleSelectOption>
          ))}
        </SimpleSelect>

        {/* Center: Endpoint selector */}
        <div css={{ flex: 1, display: 'flex', justifyContent: 'center', maxWidth: 400, margin: '0 auto' }}>
          <SimpleSelect
            id="playground-endpoint-select"
            componentId="mlflow.playground.endpoint-select"
            value={config.selectedEndpoint?.endpoint_id ?? ''}
            onChange={({ target }) => selectEndpoint(target.value || null)}
            placeholder={isLoadingEndpoints ? 'Loading...' : 'Select an endpoint'}
            disabled={isLoadingEndpoints}
            css={{ width: '100%' }}
            contentProps={{ matchTriggerWidth: true, maxHeight: 300 }}
          >
            {endpoints.map((endpoint) => (
              <SimpleSelectOption key={endpoint.endpoint_id} value={endpoint.endpoint_id}>
                {endpoint.name ?? endpoint.endpoint_id}
              </SimpleSelectOption>
            ))}
          </SimpleSelect>
        </div>

        {/* Right: Settings + Clear */}
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, flexShrink: 0 }}>
          <Popover.Root
            componentId="mlflow.playground.settings-popover"
            open={settingsOpen}
            onOpenChange={setSettingsOpen}
          >
            <Popover.Trigger asChild>
              <Button componentId="mlflow.playground.settings" icon={<GearIcon />} aria-label="Settings" />
            </Popover.Trigger>
            <Popover.Content css={{ padding: theme.spacing.md, width: 280 }} align="end">
              <Popover.Arrow />
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
                <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography.Text bold>
                    <FormattedMessage defaultMessage="Parameters" description="Playground settings popover title" />
                  </Typography.Text>
                  <Popover.Close asChild>
                    <Button
                      componentId="mlflow.playground.settings-close"
                      icon={<CloseIcon />}
                      size="small"
                      aria-label="Close settings"
                    />
                  </Popover.Close>
                </div>
                <Typography.Text color="secondary" size="sm">
                  <FormattedMessage
                    defaultMessage="These parameters control the generation process of the model."
                    description="Playground settings description"
                  />
                </Typography.Text>
                {/* Temperature */}
                <div>
                  <FormUI.Label htmlFor="playground-temperature">
                    <FormattedMessage defaultMessage="Temperature" description="Playground temperature label" />
                  </FormUI.Label>
                  <Input
                    componentId="mlflow.playground.temperature"
                    id="playground-temperature"
                    type="number"
                    value={config.temperature}
                    onChange={(e) => {
                      const val = parseFloat(e.target.value);
                      if (!isNaN(val) && val >= 0 && val <= 2) setTemperature(val);
                    }}
                    min={0}
                    max={2}
                    step={0.1}
                  />
                </div>
                {/* Max tokens */}
                <div>
                  <FormUI.Label htmlFor="playground-max-tokens">
                    <FormattedMessage defaultMessage="Max tokens" description="Playground max tokens label" />
                  </FormUI.Label>
                  <Input
                    componentId="mlflow.playground.max-tokens"
                    id="playground-max-tokens"
                    type="number"
                    value={config.maxTokens ?? ''}
                    onChange={(e) => {
                      const val = e.target.value ? parseInt(e.target.value, 10) : undefined;
                      setMaxTokens(val && val > 0 ? val : undefined);
                    }}
                    placeholder="Auto"
                    min={1}
                  />
                </div>
              </div>
            </Popover.Content>
          </Popover.Root>

          {messages.length > 0 && (
            <Button componentId="mlflow.playground.clear" size="small" icon={<RefreshIcon />} onClick={clearChat}>
              <FormattedMessage defaultMessage="Clear chat" description="Clear playground chat button" />
            </Button>
          )}
        </div>
      </div>

      {/* Chat area (full width now, no sidebar) */}
      <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
        {/* Messages */}
        <div
          css={{
            flex: 1,
            overflowY: 'auto',
            padding: theme.spacing.lg,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: messages.length === 0 ? 'center' : 'flex-start',
            alignItems: messages.length === 0 ? 'center' : 'stretch',
            maxWidth: 900,
            width: '100%',
            margin: '0 auto',
          }}
        >
          {messages.length === 0 && (
            <div
              css={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: theme.spacing.md,
                padding: theme.spacing.lg * 2,
              }}
            >
              <SparkleDoubleIcon color="ai" css={{ fontSize: 48, opacity: 0.5 }} />
              <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeMd, textAlign: 'center' }}>
                {hasEndpoint ? (
                  <FormattedMessage
                    defaultMessage="Send a message to start chatting with this endpoint."
                    description="Playground empty state with endpoint selected"
                  />
                ) : (
                  <FormattedMessage
                    defaultMessage="Select an endpoint to start chatting."
                    description="Playground empty state without endpoint"
                  />
                )}
              </Typography.Text>
            </div>
          )}

          {messages.map((message, index) => (
            <PlaygroundMessageBubble
              key={message.id}
              message={message}
              isLastMessage={message.role === 'assistant' && index === messages.length - 1}
            />
          ))}

          <div ref={messagesEndRef} />
        </div>

        {/* Error banner */}
        {error && !messages.some((m) => m.isError) && (
          <div
            css={{
              padding: `${theme.spacing.sm}px ${theme.spacing.lg}px`,
              backgroundColor: theme.colors.backgroundValidationDanger,
              color: theme.colors.textValidationDanger,
              fontSize: theme.typography.fontSizeSm,
            }}
          >
            {error}
          </div>
        )}

        {/* Input area */}
        <div
          css={{
            padding: `${theme.spacing.sm}px ${theme.spacing.md}px ${theme.spacing.md}px`,
            flexShrink: 0,
            maxWidth: 900,
            width: '100%',
            margin: '0 auto',
          }}
        >
          {/* Collapsible system prompt */}
          {!showSystemPrompt ? (
            <Button
              componentId="mlflow.playground.add-system-prompt"
              type="link"
              icon={<PlusIcon />}
              onClick={handleOpenSystemPrompt}
              css={{ marginBottom: theme.spacing.xs, padding: 0 }}
            >
              <FormattedMessage defaultMessage="Add system prompt" description="Playground add system prompt button" />
            </Button>
          ) : (
            <div
              css={{
                marginBottom: theme.spacing.sm,
                border: `1px solid ${theme.colors.borderDecorative}`,
                borderRadius: theme.borders.borderRadiusMd,
                padding: theme.spacing.sm,
                backgroundColor: theme.colors.backgroundPrimary,
              }}
            >
              <Typography.Text bold size="sm" css={{ marginBottom: theme.spacing.xs, display: 'block' }}>
                <FormattedMessage defaultMessage="System Prompt:" description="Playground system prompt label" />
              </Typography.Text>
              <Input.TextArea
                componentId="mlflow.playground.system-prompt"
                value={systemPromptDraft}
                onChange={(e) => setSystemPromptDraft(e.target.value)}
                placeholder="Optionally override the system prompt."
                rows={2}
                css={{ resize: 'vertical', width: '100%' }}
              />
              <div
                css={{
                  display: 'flex',
                  justifyContent: 'flex-end',
                  gap: theme.spacing.xs,
                  marginTop: theme.spacing.xs,
                }}
              >
                <Button
                  componentId="mlflow.playground.system-prompt-reset"
                  type="link"
                  size="small"
                  onClick={handleResetSystemPrompt}
                >
                  <FormattedMessage defaultMessage="Reset" description="Reset system prompt button" />
                </Button>
                <Button
                  componentId="mlflow.playground.system-prompt-cancel"
                  size="small"
                  onClick={handleCancelSystemPrompt}
                >
                  <FormattedMessage defaultMessage="Cancel" description="Cancel system prompt button" />
                </Button>
                <Button
                  componentId="mlflow.playground.system-prompt-save"
                  type="primary"
                  size="small"
                  onClick={handleSaveSystemPrompt}
                >
                  <FormattedMessage defaultMessage="Save" description="Save system prompt button" />
                </Button>
              </div>
            </div>
          )}

          {/* Chat input box */}
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.borders.borderRadiusMd,
              padding: theme.spacing.sm,
              backgroundColor: theme.colors.backgroundPrimary,
              opacity: hasEndpoint ? 1 : 0.5,
            }}
          >
            <div css={{ display: 'flex', alignItems: 'flex-end' }}>
              <textarea
                ref={textareaRef}
                placeholder="Start typing ..."
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={!hasEndpoint}
                rows={1}
                css={{
                  flex: 1,
                  border: 'none',
                  outline: 'none',
                  backgroundColor: 'transparent',
                  fontSize: theme.typography.fontSizeBase,
                  color: theme.colors.textPrimary,
                  padding: theme.spacing.xs,
                  resize: 'none',
                  overflowX: 'hidden',
                  overflowY: 'auto',
                  fontFamily: 'inherit',
                  lineHeight: 'inherit',
                  maxHeight: 150,
                  '&::placeholder': { color: theme.colors.textPlaceholder },
                  '&:focus': { border: 'none', outline: 'none' },
                }}
              />
              <Button
                componentId="mlflow.playground.send"
                onClick={isStreaming ? stopStreaming : handleSend}
                disabled={!isStreaming && (!inputValue.trim() || !hasEndpoint)}
                icon={isStreaming ? <StopIcon /> : <SendIcon />}
                aria-label={isStreaming ? 'Stop' : 'Send message'}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PlaygroundPage;
