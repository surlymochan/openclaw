import { beforeEach, describe, expect, it, vi } from "vitest";
import { createTestAdmittedRunContext } from "../../admitted-run-context.test-support.js";
import type { PreparedModelRuntimeSnapshot } from "../../prepared-model-runtime.js";
import { prepareEmbeddedRunRuntime } from "./runtime.js";

// Transport preparation, lease acquisition, and generation scopes have their own
// suites; here they only hand back controllable fixtures so the observed
// behavior is lease disposal.
const prepareEmbeddedRunTransportMock = vi.hoisted(() => vi.fn());
const getPreparedModelRuntimeSnapshotMock = vi.hoisted(() => vi.fn());
const acquirePreparedModelRuntimeSnapshotMock = vi.hoisted(() => vi.fn());

vi.mock("./runtime-preparation.js", () => ({
  prepareEmbeddedRunTransport: prepareEmbeddedRunTransportMock,
}));
vi.mock("../../prepared-model-runtime.js", () => ({
  getPreparedModelRuntimeSnapshot: getPreparedModelRuntimeSnapshotMock,
  acquirePreparedModelRuntimeSnapshot: acquirePreparedModelRuntimeSnapshotMock,
}));
vi.mock("../../prepared-model-runtime-generation-scope.js", () => ({
  withPreparedModelRuntimePluginGenerationScope: (_generation: unknown, run: () => unknown) => run(),
}));
vi.mock("../../../plugins/runtime/generation-scope.js", () => ({
  withPluginRuntimeGenerationScope: (_snapshot: unknown, run: () => unknown) => run(),
}));
vi.mock("../../../config/model-provider-config.js", () => ({
  resolveMergedModelProviderConfig: (config: unknown) => ({ config }),
}));
vi.mock("../../provider-local-service.js", () => ({
  getModelProviderLocalService: () => undefined,
}));
vi.mock("../../provider-local-service-reconcile.js", () => ({
  getModelProviderLocalServiceReconciler: () => undefined,
}));
vi.mock("../../../plugins/provider-runtime.js", () => ({
  providerOwnsDynamicModelPreparation: () => false,
}));
vi.mock("../../runtime-plan/credential-scoped-model.js", () => ({
  providerUsesCredentialScopedModelMetadata: () => false,
}));

const RUN_ID = "run:transport-lease";
const INITIAL_CONFIG = { generation: "initial" };
const REFRESHED_CONFIG = { generation: "refreshed" };
const METADATA_SNAPSHOT = { id: "metadata" };

type TransportFixture = {
  config: unknown;
  lastProfileId?: string;
};

function createTransportFixture(fixture: TransportFixture) {
  return {
    stopRuntimeAuthRefreshTimer: vi.fn(),
    getApiKeyInfo: () => ({ apiKey: "fixture-key" }),
    advanceAttemptAuthProfile: () => {},
    maybeRefreshRuntimeAuthForAuthError: async () => false,
    resolveRunAttemptAuthProfileStore: () => undefined,
    authStorage: undefined,
    modelRegistry: undefined,
    attemptAuthProfileStore: undefined,
    profileFailureStore: undefined,
    profileCandidates: [],
    pluginHarnessOwnsTransport: false,
    nativeModelOwned: false,
    config: fixture.config,
    snapshot: () => ({
      effectiveModel: {
        id: "fixture-model",
        api: "openai-responses",
        baseUrl: "https://api.openai.com/v1",
      },
      thinkLevel: "off",
      activePreparedAuthPlan: {
        credentialSource: { kind: "direct", evidence: "provider-config", authorization: "declared" },
      },
      lastProfileId: fixture.lastProfileId,
      runtimeAuthState: undefined,
    }),
  };
}

function publishedSnapshot(config: unknown): PreparedModelRuntimeSnapshot {
  return { config, metadataSnapshot: METADATA_SNAPSHOT } as unknown as PreparedModelRuntimeSnapshot;
}

function prepareRuntime() {
  return prepareEmbeddedRunRuntime({
    assertCurrent: () => {},
    runParams: {
      runId: RUN_ID,
      admittedRunContext: createTestAdmittedRunContext(RUN_ID),
      sessionId: "lease-session",
      sessionKey: "agent:main:lease-session",
      agentId: "main",
      prompt: "Reply briefly.",
      workspaceDir: "/tmp/lease-session",
      timeoutMs: 5_000,
      config: INITIAL_CONFIG,
    },
    provider: "openai",
    modelId: "fixture-model",
    agentDir: "/tmp/lease-session/agent",
    workspaceDir: "/tmp/lease-session",
    globalLane: "test",
    hookRunner: undefined,
    hookContext: { sessionId: "lease-session", workspaceDir: "/tmp/lease-session" },
    markStartupStage: () => {},
    notifyExecutionPhase: () => {},
    fallbackConfigured: false,
    preparedModelRuntime: publishedSnapshot(INITIAL_CONFIG),
  } as unknown as Parameters<typeof prepareEmbeddedRunRuntime>[0]);
}

/**
 * Refreshes onto a second transport so the run retains one lease. A truthy
 * `candidateLastProfileId` fails late admission and discards the candidate,
 * which disposes its lease in the refresh fallback instead.
 */
async function prepareRuntimeWithLease(params: {
  dispose: () => Promise<void>;
  candidateLastProfileId?: string;
}) {
  const disposeLease = vi.fn(params.dispose);
  prepareEmbeddedRunTransportMock
    .mockResolvedValueOnce(createTransportFixture({ config: INITIAL_CONFIG }))
    .mockResolvedValueOnce(
      createTransportFixture({
        config: REFRESHED_CONFIG,
        lastProfileId: params.candidateLastProfileId,
      }),
    );
  getPreparedModelRuntimeSnapshotMock.mockReturnValue(publishedSnapshot(REFRESHED_CONFIG));
  acquirePreparedModelRuntimeSnapshotMock.mockResolvedValue({
    snapshot: publishedSnapshot(REFRESHED_CONFIG),
    pluginGeneration: { remoteCatalog: undefined, metadataSnapshot: METADATA_SNAPSHOT },
    [Symbol.asyncDispose]: disposeLease,
  });
  const runtime = await prepareRuntime();
  const transportConfigBefore = runtime.snapshot().transportConfig;
  await runtime.refreshTransportForConnectionError();
  return { runtime, disposeLease, transportConfigBefore };
}

describe("embedded run transport lease disposal", () => {
  beforeEach(() => {
    prepareEmbeddedRunTransportMock.mockReset();
    getPreparedModelRuntimeSnapshotMock.mockReset();
    acquirePreparedModelRuntimeSnapshotMock.mockReset();
  });

  it("resolves releaseTransport when a retained lease disposal rejects", async () => {
    const { runtime, disposeLease } = await prepareRuntimeWithLease({
      dispose: async () => {
        throw new Error("lease dispose failed");
      },
    });
    await expect(runtime.releaseTransport()).resolves.toBeUndefined();
    expect(disposeLease).toHaveBeenCalledTimes(1);
  });

  it("settles releaseTransport within the bound when a lease disposal never settles", async () => {
    const { runtime } = await prepareRuntimeWithLease({ dispose: () => new Promise<void>(() => {}) });
    // Real time: faked timers do not reach the run module's timer realm, and an
    // unbounded dispose would otherwise only fail by hanging until test timeout.
    const startedAtMs = Date.now();
    let deadline: ReturnType<typeof setTimeout> | undefined;
    const outcome = await Promise.race([
      runtime.releaseTransport().then(() => "settled" as const),
      new Promise<"still-pending">((resolve) => {
        deadline = setTimeout(() => resolve("still-pending"), 60_000);
      }),
    ]);
    clearTimeout(deadline);
    expect(outcome).toBe("settled");
    expect(Date.now() - startedAtMs).toBeLessThan(60_000);
  });

  it("keeps the old transport when a discarded refresh candidate's lease disposal rejects", async () => {
    const { runtime, disposeLease, transportConfigBefore } = await prepareRuntimeWithLease({
      dispose: async () => {
        throw new Error("candidate lease dispose failed");
      },
      candidateLastProfileId: "openai:refreshed",
    });
    expect(disposeLease).toHaveBeenCalledTimes(1);
    expect(runtime.snapshot().transportConfig).toBe(transportConfigBefore);
    await expect(runtime.releaseTransport()).resolves.toBeUndefined();
  });
});
