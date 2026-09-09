Feature: verify

  @VF-01 @build
  Scenario: A receipt fetched by its kappa from /v1/receipts returns the stored document whose signature and kappa verify.
    Given a chat completion that carried an x-hologram-receipt header
    When GET /v1/receipts/{kappa} is requested
    Then the document's Ed25519 signature verifies over its canonical bytes and its kappa matches the sealed bytes

  @VF-02 @build
  Scenario: Verifying a receipt whose engine sealed no replayable answer record reports not verified with integrity true and never claims a replay.
    Given a receipt produced by the echo engine
    When POST /v1/receipts/{kappa}/verify is requested
    Then the verdict has integrity true, verified false, and a reason naming the missing record
