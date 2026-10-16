# Live Gemini adapter integration handoff

The live-provider developer retains ownership of Gemini credential, model,
transport, framing, and schema verification. The MVP does not change
`printgpt_generation.gemini` and does not require Gemini for the offline demo.

The integration seam is `GraphDependencies.generator` and
`GraphDependencies.critic`. A live run should replace only the deterministic
generator and critic used by `printgpt_mvp.pipeline._offline_dependencies`.
It must continue to use `TrackAValidator`, the canonical page and `MaskBundle`
created by the MVP, and the existing bounded LangGraph workflow. Approval and
export must still receive only a ranked attempt's
`sanitized_new_ink_mask`; raw Gemini images are diagnostic and never printable.

Live verification should record:

- the exact model IDs and SDK version;
- whether 3:4 padding/cropping preserves registration on real marked A4 pages;
- source-retention, estimated shift, and extraction metrics for both candidates;
- critic schema compliance and whether the critic saw the sanitized preview;
- provider failures and bounded retry counts;
- confirmation that a failed live run leaves the deterministic offline demo
  usable and creates no printable output.

If live testing reveals an integration defect outside the adapter, reproduce it
with a deterministic or recorded test fixture before changing the shared
pipeline. Do not weaken Track A validation, approval, or mask-dimension checks
to accommodate provider output.
