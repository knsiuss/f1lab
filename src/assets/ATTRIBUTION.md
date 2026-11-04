# Image attribution

Local image assets used by the dashboard. Refresh them with
`python tools/download_images.py`.

## Driver portraits (`drivers/`)

- Most portraits are Formula 1 media-CDN images (media.formula1.com),
  reached through the `image_url` values already recorded in
  `src/config.py` (`DRIVER_PROFILES`). © Formula One Group. Used here
  for non-commercial display only; rights remain with the copyright
  holder. No image is modified beyond resizing by the CDN transform.
- `andrea_kimi_antonelli.jpg`, `ryo_hirakawa.jpg`, `arvid_lindblad.jpg`:
  Wikipedia page thumbnails (CC BY-SA). Sources:
  https://en.wikipedia.org/wiki/Kimi_Antonelli,
  https://en.wikipedia.org/wiki/Ryo_Hirakawa,
  https://en.wikipedia.org/wiki/Arvid_Lindblad

## Country flags (`flags/`)

- Downloaded from flagcdn (https://flagcdn.com), whose flag set is sourced
  from Wikimedia with public-domain country flags. Displayed at small
  size next to Grand Prix names.

## Team identity

Team logos are trademarks and are intentionally not bundled. Teams are
identified by their official livery colours (from the OpenF1 results
feed) and names.
