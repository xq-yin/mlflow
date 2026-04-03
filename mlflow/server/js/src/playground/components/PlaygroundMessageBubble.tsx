import { useState } from 'react';
import { Button, SparkleIcon, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { GenAIMarkdownRenderer } from '../../shared/web-shared/genai-markdown-renderer';
import { useCopyController } from '../../shared/web-shared/snippet/hooks/useCopyController';
import type { PlaygroundMessage } from '../hooks/usePlaygroundChat';

const PULSE_ANIMATION = {
  '0%, 100%': { transform: 'scale(1)' },
  '50%': { transform: 'scale(1.3)' },
};

const DOTS_ANIMATION = {
  '0%': { content: '""' },
  '33%': { content: '"."' },
  '66%': { content: '".."' },
  '100%': { content: '"..."' },
};

export const PlaygroundMessageBubble = ({
  message,
  isLastMessage,
}: {
  message: PlaygroundMessage;
  isLastMessage: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const isUser = message.role === 'user';
  const [isHovered, setIsHovered] = useState(false);
  const { actionIcon: copyIcon, tooltipMessage: copyTooltip, copy: handleCopy } = useCopyController(message.content);

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: isUser ? 'flex-end' : 'flex-start',
        marginBottom: isUser ? theme.spacing.md : 0,
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div
        css={{
          maxWidth: isUser ? '85%' : '100%',
          padding: `${theme.spacing.md}px ${theme.spacing.md}px ${isUser ? theme.spacing.md : 0}px ${theme.spacing.md}px`,
          borderRadius: theme.borders.borderRadiusLg,
          backgroundColor: isUser ? theme.colors.backgroundSecondary : 'transparent',
          color: message.isError ? theme.colors.textValidationDanger : theme.colors.textPrimary,
        }}
      >
        {isUser ? (
          <Typography.Text css={{ whiteSpace: 'pre-wrap' }}>{message.content}</Typography.Text>
        ) : message.isError ? (
          <Typography.Text color="error" css={{ whiteSpace: 'pre-wrap' }}>
            {message.content}
          </Typography.Text>
        ) : (
          <GenAIMarkdownRenderer>{message.content}</GenAIMarkdownRenderer>
        )}
        {message.isStreaming && (
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.sm,
              marginTop: theme.spacing.sm,
            }}
          >
            <SparkleIcon
              color="ai"
              css={{
                fontSize: 16,
                animation: 'pulse 1.5s ease-in-out infinite',
                '@keyframes pulse': PULSE_ANIMATION,
              }}
            />
            <span
              css={{
                fontSize: theme.typography.fontSizeBase,
                color: theme.colors.textSecondary,
                '&::after': {
                  content: '"..."',
                  animation: 'dots 1.5s steps(3, end) infinite',
                  display: 'inline-block',
                  width: '1.2em',
                },
                '@keyframes dots': DOTS_ANIMATION,
              }}
            >
              Processing
            </span>
          </div>
        )}
      </div>

      {!isUser && !message.isStreaming && !message.isError && message.content && (
        <div
          css={{
            display: 'flex',
            gap: theme.spacing.xs,
            paddingLeft: theme.spacing.md,
            opacity: isLastMessage || isHovered ? 1 : 0,
            pointerEvents: isLastMessage || isHovered ? 'auto' : 'none',
            transition: 'opacity 0.2s ease',
          }}
        >
          <Tooltip componentId="mlflow.playground.message.copy.tooltip" content={copyTooltip}>
            <Button componentId="mlflow.playground.message.copy" size="small" icon={copyIcon} onClick={handleCopy} />
          </Tooltip>
        </div>
      )}
    </div>
  );
};
