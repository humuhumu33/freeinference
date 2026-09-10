Feature: webgpu

  @WG-01 @build
  Scenario: The daemon serves the pinned Hologram Q WebGPU engine snapshot under /q with its hash list, and the wasm with the wasm media type.
    Given a running daemon
    When /q/SNAPSHOT.txt, /q/core/q-brain-fast.mjs and /q/pkg/holospaces_web_bg.wasm are requested
    Then each is served, the hash list names every module, and the wasm carries application/wasm

  @WG-02 @build
  Scenario: The Playground offers the WebGPU models only when the browser has WebGPU, seals each browser answer under Hologram Q's receipt, and hands it to the daemon through /v1/webgpu/seal.
    Given the Playground page
    When its script is read
    Then the WebGPU options are gated on navigator.gpu, answers are sealed with the engine's buildReceipt, and posted to /v1/webgpu/seal

  @WG-03 @build
  Scenario: An answer the browser engine sealed is stored as its Q receipt, its answer bytes and a memo, only when the receipt's did:holo re-derives from its body; the same prompt is then served from the receipt to the Playground and to any OpenAI client with no execution, and the receipt's integrity verifies on the daemon.
    Given a daemon and a Q receipt sealed for a prompt
    When the receipt is posted to /v1/webgpu/seal, then the prompt is looked up and sent through the OpenAI surface
    Then a forged receipt is refused, the genuine one is stored as three objects, both lookups return the stored answer with the receipt and a reuse header, and verify reports integrity

  @WG-04 @build
  Scenario: As shipped, the daemon's engine is the browser's: /v1/models lists the WebGPU models, and a prompt no receipt answers is refused with where compute runs, never executed or echoed on the daemon.
    Given a daemon built exactly as freeinference serve builds it
    When /v1/models is read and a prompt no receipt answers is sent
    Then the WebGPU models are listed and the prompt is refused naming /playground, with nothing sealed
