Feature: console

  @CO-01 @build
  Scenario: The daemon serves a dashboard at /dashboard that lists the sealed receipts on this machine.
    Given a running daemon
    When /dashboard is requested
    Then the page is served and reads the receipts from the local object registry

  @CO-02 @build
  Scenario: The daemon serves a playground at /playground that sends prompts to the local OpenAI surface and shows each answer's fingerprint and receipt.
    Given a running daemon
    When /playground is requested
    Then the page is served and posts to /v1/chat/completions, showing system_fingerprint and the receipt header
