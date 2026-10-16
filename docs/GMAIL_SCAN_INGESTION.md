# Gmail scan ingestion

PrintGPT can import pages sent by the Canon imageRUNNER 2425i "Scan and Send"
function without anyone opening Gmail. The app watches the recipient inbox
with **read-only** Gmail access. It downloads the newest scanner attachment and
passes it to the existing Scanner A4 normalization. The corrected A4 preview
then appears for confirmation before masking and generation. Manual upload
remains available.

```text
Start waiting for scan -> Canon Scan and Send -> Gmail -> newest scanner email found
  -> attachment validated and saved atomically -> Scanner A4 normalization
  -> corrected A4 preview -> user confirms -> masking and generation
```

## How a scan is chosen

Each poll (every 4 s by default) searches narrowly:

```text
from:<sender> to:<recipient> has:attachment newer_than:7d
```

It fetches at most 10 messages and re-checks them exactly. A message counts
only if its `From` address equals the configured sender, a `To`/`Cc`/
`Delivered-To` address equals the recipient, the optional subject filter
matches, and it has a supported attachment.

- **Only the newest matching message is considered.** Older messages are never
  imported automatically. Read/unread state is ignored.
- If the newest message was already imported, the session keeps waiting for a
  new scan.
- Within that message a PDF is preferred over image alternatives. If more than
  one equally preferred attachment remains (for example two PDFs), the app
  shows them and the user chooses.
- There is no "received after the session started" window. The first session
  after setup therefore imports the newest existing scan if it was never
  imported. This is safe because the preview must be confirmed, and every
  later session waits for a genuinely new email.

Supported attachments: PDF, PNG, JPEG, and TIFF. Extension and MIME type must
agree; `application/octet-stream` is accepted when the extension is supported.
The file content must also start with the matching file signature.

## Safe download and duplicate prevention

Every attachment goes through these checks:

1. It is size-limited (50 MB by default; the declared size is checked before
   download).
2. It is rejected if empty, an archive, or an executable.
3. Its content must match its type.
4. The filename is sanitized to a plain basename, and path traversal is
   rejected.
5. It is written to a temporary file, flushed, fsynced, and renamed
   atomically into `artifacts/email_scans/email-<sha256 prefix>/input/`.
6. The saved file is read back and its hash re-checked. A PDF must contain at
   least one page.

The original bytes are never altered. Downloads never go into source
directories.

`processed.json` in the state directory records the message ID, attachment
ID, SHA-256, run ID, and session. All writes run under a cross-process lock,
so Streamlit and the MCP server can share the file.

- A repeated message returns `already_processed`.
- A different message containing an identical file also returns
  `already_processed`, and is recorded so it is never downloaded again.
- A rejected attachment is recorded as `rejected`, so it does not block later
  sessions.

## Google Cloud setup (once, free)

1. Open <https://console.cloud.google.com/> while signed in as the recipient
   account, and create or choose a project. No billing is needed.
2. **APIs & Services → Library → Gmail API → Enable.**
3. **Google Auth Platform → Branding:** set an app name and support email.
   For an `@gmail.com` account the audience is **External**. Leave the app in
   **Testing**; do not publish it.
4. **Audience → Test users → Add users:** add the recipient Gmail address.
   While in testing, only listed test users can consent. Google may also
   expire refresh tokens after 7 days, which means reconnecting weekly.
5. **Data Access → Add or remove scopes:** add only
   `https://www.googleapis.com/auth/gmail.readonly`.
6. **Clients → Create client → Application type: Desktop app.** Download the
   JSON and store it outside the repository, for example
   `%LOCALAPPDATA%\PrintGPT\client_secret.json`.

## Local configuration

Copy `.env.example` to `.env` (ignored by Git) and fill in:

| Variable | Meaning |
| --- | --- |
| `PRINTGPT_GMAIL_ACCOUNT` | Recipient inbox that is monitored |
| `PRINTGPT_SCAN_SENDER` | Address the Canon sends from (exact match) |
| `PRINTGPT_GMAIL_CREDENTIALS_PATH` | Absolute path to the desktop-client JSON |
| `PRINTGPT_GMAIL_TOKEN_PATH` | Optional; default `%LOCALAPPDATA%\PrintGPT\gmail_token.json` |
| `PRINTGPT_SCAN_SUBJECT_FILTER` | Optional subject prefix, e.g. `PrintGPT Scan` |
| `PRINTGPT_SCAN_LOOKBACK_DAYS` | Search window, default 7 |
| `PRINTGPT_SCAN_MAX_ATTACHMENT_MB` | Default 50 |
| `PRINTGPT_SCAN_POLL_INTERVAL_S` | Default 4 |
| `PRINTGPT_SCAN_TIMEOUT_S` | Default 180 (three minutes) |
| `PRINTGPT_SCAN_STATE_DIR` | Default `%LOCALAPPDATA%\PrintGPT\email_ingestion` |
| `PRINTGPT_SCAN_RUNS_ROOT` | Default `artifacts/email_scans` |

Environment variables override `.env`. Install the dependencies:

```powershell
python -m pip install -e ".[app,gmail]"
```

## Connecting and the read-only scope

```powershell
python -m printgpt_mvp.email_ingestion connect
```

Alternatively, choose **Scanner email (Gmail) → Connect Gmail (read-only)** in
the app. The browser opens Google consent, and you sign in as the recipient
account.

- Only `gmail.readonly` is requested; the app cannot send, modify, label,
  trash, or delete mail.
- No password, app password, or token is typed into PrintGPT. The token is
  stored at the token path and refreshed automatically afterwards.
- A token with any other scope is ignored. A token belonging to a different
  account is refused.
- On Windows the token file relies on the per-user `%LOCALAPPDATA%` folder's
  permissions. Windows Credential Manager is not used in this MVP.

## Using it

```powershell
python -m streamlit run src/printgpt_mvp/streamlit_app.py
```

1. Choose the input mode (normally **Scanner A4**) and orientation.
2. Choose **Scan source → Scanner email (Gmail)** and press **Start waiting for scan**.
3. At the Canon: place the page, choose **Scan and Send**, and send to the
   recipient address. Note the subject the Canon uses; you can set it as the
   optional filter.
4. The page shows *Waiting for scanner email → Email found → Downloading →
   Validating → Normalizing A4*. Polling runs in a Streamlit fragment, so the
   page never blocks. **Cancel waiting** stops it.
5. The corrected canonical A4 preview appears. Confirm it to enable masking and
   generation.

ArUco marker detection runs only if you explicitly chose **Marked page**. Waiting
ends after the timeout. Session state is stored on disk, so reruns and reloads
keep it.

For a command-line check that stops before generation:

```powershell
python -m printgpt_mvp.email_ingestion wait
```

It imports one new scan and writes `canonical_a4_preview.png` next to the
imported file. It also prints the observed subject and confirms that a second
import of the same message returns `already_processed`.

## Logging and privacy

Logs contain only the session ID, Gmail message ID, attachment filename, MIME
type, size, hash, timestamps, and result. They never contain tokens, client
secrets, message bodies, or subjects/addresses of unrelated mail. Message bodies
returned by the API are never read or stored.

## Revoking access

- Locally: `python -m printgpt_mvp.email_ingestion revoke` deletes the token.
- At Google: <https://myaccount.google.com/permissions> → select the app →
  **Remove access**.
- Delete the OAuth client in Google Cloud if it is no longer needed.

## Tests

`python -m pytest tests/email_ingestion` uses a fake Gmail client and fixture
messages only. The real-inbox check is opt-in:

```powershell
$env:PRINTGPT_LIVE_GMAIL = "1"; python -m pytest tests/email_ingestion/test_email_live_gmail.py
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `access_denied` / "app not verified" | Add the recipient as a test user (setup step 4) |
| "OAuth client file not found / not Desktop app" | Check `PRINTGPT_GMAIL_CREDENTIALS_PATH`; create a **Desktop app** client |
| Asked to reconnect every week | Testing-mode refresh tokens expire after 7 days; reconnect |
| "Signed in as … not the configured scan inbox" | Run `revoke`, then connect with the recipient account |
| Scan never detected | Check the Canon's sender address matches `PRINTGPT_SCAN_SENDER` exactly; remove the subject filter; mail can take a minute |
| "latest scan already imported" | Send a new scan; old ones are never re-imported |
| No supported attachment | Set the Canon to send PDF (preferred), TIFF, or JPEG, not a ZIP |
| Several attachments | Choose one in the app, or send one page per scan |
| Timed out | Start waiting again; raise `PRINTGPT_SCAN_TIMEOUT_S` if mail is slow |
