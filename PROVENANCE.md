# Provenance - non-code assets

Record of where this repository's image assets came from, per CireSnave's provenance rule
(`C:\Projects\CIRESNAVE-EXPECTATIONS.md` section 6.4a, 2026-09-17). Measured at `origin/main` `962e5c39`, 2026-10-24.
Documentation only: no asset was changed.

| file | maker / tool | date | licence |
|---|---|---|---|
| `vulkan.jpeg` (sha256 `d8419f4d0908b5aaae54371107400aa8a2438b8a5d82b1dd60817c73d86d9028`, 200340 bytes) | Generated for this project by an AI image tool. CireSnave (2026-10-24, verbatim): "Vulkane's logo was custom made for us by an LLM that said it claimed no rights to it.  I don't remember which one." The file's embedded C2PA manifest (see below) states it was created with Microsoft's Image Creator from Designer. | Embedded timestamp 2025-07-19T11:57:14Z; first committed 2026-04-07 (`7f04b9213b7b3dc03731bd6463a5089f27589805`, "Initial commit: spock Vulkan bindings generator") | no rights claimed by the generating tool per CireSnave; the output terms of the service were not checked here (pending CireSnave, decision 267) |

## What the embedded C2PA manifest says

Read-only parse of the JPEG's APP11 (JUMBF) segment (Python `-I`, no third-party libraries; the file was not modified).
It holds a C2PA manifest store (`urn:uuid:518f5bfb-052e-404a-b5a4-882e4fe6793e`) containing:

- `c2pa.actions` assertion: action `c2pa.created`, `softwareAgent` = `Image Creator from Designer`,
  `when` = `2025-07-19T11:57:14Z`, `digitalSourceType` = `http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia`.
- `c2pa.hash.boxes` assertion (sha256 hashes over the JPEG segments).
- Claim (`c2pa.claim`): `claim_generator` = `Microsoft_Responsible_AI/1.0`; `claim_generator_info` name =
  `Microsoft Responsible AI Image Provenance`, version `1.0`; format `image/jpeg`.
- Signature (`c2pa.signature`): X.509 chain whose leaf subject is `Microsoft Corporation` (Redmond, Washington), issued by
  `Microsoft SCD Claimants RSA CA` (validity 2024-10-24 to 2025-10-24), rooted at `Microsoft Supply Chain RSA Root CA 2022`;
  RFC 3161 timestamp token from DigiCert (`DigiCert SHA256 RSA4096 Timestamp Responder 2025 1`), time 2025-07-19T11:57:13Z.

**Limit:** this was a structural read of the strings and fields only. The signature and the hash assertions were NOT
cryptographically validated (no C2PA validator was run), so the manifest is evidence of what the file claims about itself,
not a verified attestation. Neither the manifest nor this file establishes the service's output terms.

**Method:** first-add commit found with `git log --follow --diff-filter=A -- vulkan.jpeg` (one result, above); sha256 taken
from `git show origin/main:vulkan.jpeg`; `origin/main` read at `962e5c395ebd0bcc5d028f6da4f619f765599c4e`. The declared licence of the repository's code is
`MIT OR Apache-2.0` (`vulkane/Cargo.toml`; `LICENSE-MIT`, `LICENSE-APACHE`); that declaration does not itself license this image.
