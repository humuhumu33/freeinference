Feature: receipts

  @RC-01 @build
  Scenario: A stock OpenAI client receives a system fingerprint equal to the resolved model kappa joined with the engine kappa.
    Given a daemon with a model imported into the local catalog
    When a chat completion is requested for that model through the OpenAI surface
    Then system_fingerprint equals the model kappa, a semicolon, and the engine kappa

  @RC-02 @build
  Scenario: The receipt named in the response header resolves to a stored object whose kappa matches its bytes and whose signature verifies.
    Given a chat completion that carried an x-hologram-receipt header
    When the named object is fetched from the registry and re-hashed
    Then the hash equals the header value and the Ed25519 signature verifies over the canonical bytes

  @RC-03 @build
  Scenario: A response for a model that does not resolve in the catalog carries no fingerprint and no receipt header.
    Given a daemon whose catalog does not contain the requested model
    When a chat completion is requested for that model
    Then the response has no system_fingerprint and no x-hologram-receipt header

  @RC-04 @build
  Scenario: Every module registers and the merged router boots without a route collision.
    Given the built in modules and every freeinference module
    When the application state is built
    Then startup succeeds and the merged router serves the health route

  @RC-05 @build
  Scenario: A repeated prompt for the same model and parameters is served from its stored receipt and answer bytes with no engine run, carrying the original receipt kappa and a reuse header.
    Given a daemon that has answered a prompt once and sealed a receipt
    When the same prompt is sent again for the same model and parameters
    Then the response carries the same receipt kappa, an x-hologram-reuse header, the same content, and no new receipt exists

  @RC-06 @build
  Scenario: The memo, receipt and answer objects carried to a daemon that has no model serve the same answer there when the model is asked for by kappa, with no engine run, and the receipt's integrity verifies on that daemon.
    Given a second daemon with an empty catalog
    When the memo, receipt and answer objects are posted to its registry and the prompt is sent naming the model by kappa
    Then the response carries the original receipt kappa, a reuse header, the same content, no receipt was sealed there, and the receipt's integrity verifies on that daemon
