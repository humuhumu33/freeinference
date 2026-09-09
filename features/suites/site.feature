Feature: site

  @SI-01 @build
  Scenario: The homepage contains no hyphen or dash in any visible text, code samples included.
    Given the built homepage at site/index.html
    When every visible text node is scanned, code samples included
    Then no hyphen minus, hyphen, dash, or minus sign character is present

  @SI-02 @build
  Scenario: The homepage headline is exactly Free verified AI inference.
    Given the built homepage at site/index.html
    When the single h1 element is read
    Then its text equals "Free verified AI inference."

  @SI-03 @build
  Scenario: The homepage prose is under 120 words, code samples excluded.
    Given the built homepage at site/index.html
    When visible prose is counted with pre and code elements excluded
    Then the count is below 120
