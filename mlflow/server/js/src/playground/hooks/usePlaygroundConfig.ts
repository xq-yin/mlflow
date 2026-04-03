import { useState, useMemo, useCallback } from 'react';
import { useEndpointsQuery } from '../../gateway/hooks/useEndpointsQuery';
import type { Endpoint } from '../../gateway/types';

const getBaseUrl = (): string => {
  if (typeof window !== 'undefined') {
    return window.location.origin;
  }
  return 'http://localhost:5000';
};

export interface PlaygroundConfig {
  selectedEndpoint: Endpoint | null;
  systemPrompt: string;
  temperature: number;
  maxTokens: number | undefined;
  requestUrl: string;
  endpointName: string;
}

export function usePlaygroundConfig() {
  const { data: endpoints, isLoading: isLoadingEndpoints } = useEndpointsQuery();

  const [selectedEndpointId, setSelectedEndpointId] = useState<string | null>(null);
  const [systemPrompt, setSystemPrompt] = useState('');
  const [temperature, setTemperature] = useState(1.0);
  const [maxTokens, setMaxTokens] = useState<number | undefined>(undefined);

  const selectedEndpoint = useMemo(
    () => endpoints.find((e) => e.endpoint_id === selectedEndpointId) ?? null,
    [endpoints, selectedEndpointId],
  );

  const endpointName = selectedEndpoint?.name ?? selectedEndpoint?.endpoint_id ?? '';

  const requestUrl = useMemo(() => {
    if (!endpointName) return '';
    const base = getBaseUrl();
    return `${base}/gateway/${endpointName}/mlflow/invocations`;
  }, [endpointName]);

  const selectEndpoint = useCallback((endpointId: string | null) => {
    setSelectedEndpointId(endpointId);
  }, []);

  const config: PlaygroundConfig = {
    selectedEndpoint,
    systemPrompt,
    temperature,
    maxTokens,
    requestUrl,
    endpointName,
  };

  return {
    config,
    endpoints,
    isLoadingEndpoints,
    selectEndpoint,
    setSystemPrompt,
    setTemperature,
    setMaxTokens,
  };
}
