import { createMLflowRoutePath } from '../common/utils/RoutingUtils';

export enum PlaygroundPageId {
  playgroundPage = 'mlflow.playground',
}

// following same pattern as other routes files
// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
export class PlaygroundRoutePaths {
  static get playgroundPage() {
    return createMLflowRoutePath('/playground');
  }
}

// following same pattern as other routes files
// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
class PlaygroundRoutes {
  static get playgroundPageRoute() {
    return PlaygroundRoutePaths.playgroundPage;
  }
}

export default PlaygroundRoutes;
