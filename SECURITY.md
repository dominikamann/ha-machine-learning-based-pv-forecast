# Security policy

## Supported versions

Only the latest release receives fixes. Please update before reporting a problem.

## Reporting a vulnerability

Please report security problems privately via
[GitHub security advisories](https://github.com/dominikamann/ha-machine-learning-based-pv-forecast/security/advisories/new)
instead of opening a public issue. You will get an answer within a few days.

## What the integration sends out

- Only your Home Assistant location (latitude, longitude) goes to [Open-Meteo](https://open-meteo.com)
  to fetch weather data. No production data leaves your Home Assistant.
- The evcc endpoint is served by Home Assistant itself and requires a Home Assistant access token.
  Use a dedicated long-lived token for evcc and never post it in issues.
- The diagnostics download redacts the location. Check attachments before posting them anyway.
