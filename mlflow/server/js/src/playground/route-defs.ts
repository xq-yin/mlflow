import type { RouteHandle } from '../common/utils/RoutingUtils';
import { createLazyRouteElement } from '../common/utils/RoutingUtils';
import { PlaygroundPageId, PlaygroundRoutePaths } from './routes';

export const getPlaygroundRouteDefs = () => {
  return [
    {
      path: PlaygroundRoutePaths.playgroundPage,
      element: createLazyRouteElement(() => import('./pages/PlaygroundPage')),
      pageId: PlaygroundPageId.playgroundPage,
      handle: {
        getPageTitle: () => 'Playground',
        getAssistantPrompts: () => ['How do I test my gateway endpoint?', 'How do I use prompts in the playground?'],
      } satisfies RouteHandle,
    },
  ];
};
