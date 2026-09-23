# Case Study: AI Document Copilot for paperless-ngx

## What is paperless-ngx?

paperless-ngx is a free, open-source document management system. It allows you
to digitize, organize, and search your physical documents. You drop scanned
documents (PDFs, images) into a watch folder, and paperless-ngx automatically
extracts text, metadata, and performs OCR to make everything searchable. Think
of it as a self-hosted Google Drive specifically designed for documents — with
powerful tagging, full-text search, and automated organization.

This project adds AI-powered OCR, metadata extraction, and a browser chat
copilot to Paperless-ngx without patching Paperless itself.

![Chat copilot screenshot](assets/chat-demo.png)

New documents follow the normal Paperless flow. An AI service re-OCRs selected
pages and extracts metadata. The chat copilot searches Paperless full text with
keywords, reads matching documents, and returns source-backed answers; it does
not build a separate semantic or vector index.

It supports both cloud and self-hosted
models, and makes model quality visible through evaluation and tracing.

The example asks how much was spent on Google Cloud in 2026. The copilot
searches Paperless, reads matching invoices, and cites the documents behind its
answer. Jev evaluates metadata extraction and judges proposed correspondent
merges in the cleanup workflow. Chat retrieval separately uses Paperless
keyword search.



## What It Does

- OCRs imported documents with a vision model and writes the transcript back
  to Paperless.
- Extracts title, date, correspondent, summary, and structured debug output via
  a metadata LLM.
- Provides a browser chat copilot that can call tools, search the archive, read
  source text, and return source-backed answers.
- Uses the shared inference client for cloud and local models, with traces and evaluation in Arize Phoenix.

Learn more in the [deep dive](deep-dive.md) and the
[Jev evaluation report](phoenix-paperless-eval-test-report.md).
