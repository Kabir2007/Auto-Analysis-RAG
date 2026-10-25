# Auto-Analysis-RAG

Auto-Analysis-RAG is a Streamlit application for exploring user-provided CSV and Excel datasets. It combines dataset cleaning and exploratory analysis with a model-planned analysis chat that selects deterministic statistical, charting, machine-learning, and forecasting functions.

The application requires a Supabase project for accounts and persistent dataset storage. Each user signs in, uploads their own data, and supplies their own supported LLM API key if they want to use Analysis Chat. The repository includes no sample dataset and no shared/master LLM key.

## What the application does

### Data & Cleaning

- Accepts CSV and `.xlsx` uploads, up to **50 MiB** and **200,000 rows**. Multi-sheet workbooks can be loaded by selecting a worksheet.
- Shows a preview, dataset dimensions, missing-cell share, exact duplicate count, inferred column types, descriptive profile values, and data-quality flags.
- Offers a cleaning pipeline for duplicate removal, missing-value handling, and outlier flagging. Numeric outliers can optionally be winsorized to the 1st/99th percentiles.
- Saves the cleaned dataset and cleaning audit information to the user’s private cloud storage. Cleaned and model-ready encoded CSVs can be downloaded.

### Exploratory Data Analysis (EDA)

- Builds numeric and missingness summaries, top categorical frequencies, and rule-based observations.
- Displays numeric histograms, categorical frequency charts, Pearson correlation heatmaps, and a weekly trend chart when date and numeric columns are available.
- EDA results are calculated from the selected raw or cleaned dataset.

### Analysis Chat

The user prompt and available column schema go to the selected LLM. It chooses a built-in function and the relevant columns; Python computes the results. The model does not write or execute analysis code.

- **Descriptive statistics:** means, medians, sums, counts, minima/maxima, standard deviations, unique counts, grouped results, and per-entity item averages.
- **Hypothesis testing:** group comparisons, numeric correlations, categorical associations, two-group proportion comparisons, and top/bottom cohort comparisons when there is an independent outcome column. Depending on data and assumptions, the implementation uses t-tests, Mann–Whitney, ANOVA, Kruskal–Wallis, chi-square/Fisher exact, Pearson/Spearman, or a two-proportion z-test. Supported outputs include p-values, effect sizes, group summaries, confidence intervals, assumption notes, and post-hoc comparisons.
- **Supervised prediction:** classification or regression. Candidate models are selected using cross-validation on the training split, then evaluated against a held-out split and a simple baseline. Results include task-appropriate metrics, a confusion matrix or actual-versus-predicted plot, and permutation feature importance where available.
- **Time-series forecasting:** aggregates a numeric target by an inferred daily, weekly, or monthly frequency, compares candidate models on chronological splits and a naive baseline, and produces a recursive forecast. Forecast bands are approximate residual-based bands, not formal guarantees.
- **Charts:** user-requested bar, line, scatter, histogram, and box plots. The model chooses the chart specification and exact dataset columns.
- **Statistical guidance:** retrieves relevant material from [`kb/stats_guide.md`](kb/stats_guide.md).
- **Traceability:** shows the structured analysis plan, selected tool, and execution trace with each response.

Statistical significance does not establish causation. Predictive feature importance is not causal evidence. The application reports warnings and baseline performance so weak models or unsupported conclusions can be recognized.

## Data, keys, and account isolation

- Supabase Auth handles email/password registration and sign-in. Dataset metadata is stored in a Postgres table with row-level security. Original uploads, analysis-ready Parquet files, and cleaned Parquet files are stored in a private Supabase Storage bucket. The bucket and per-user policies are defined in [`supabase_setup.sql`](supabase_setup.sql).
- Each user’s objects are stored below their Supabase Auth user ID. The application sends the signed-in user’s JWT with data requests; it does not use a service-role key.
- Dataset files remain in cloud storage until the owner deletes the dataset from the app. The delete control removes its stored objects and metadata.
- Provider API keys are entered in the sidebar and used for that Streamlit session. They are not written to Supabase or environment variables. The displayed chat history is session-only; a compact prior-query/finding memory is also held in process memory, scoped to the user and dataset. Neither is saved as a conversation archive.
- Analysis prompts include the user’s question, column names/types, relevant dataset profile context, and deterministic tool results. Profile context can include numeric summaries, date ranges, quality flags, and frequent category values. The full dataset rows are processed by Python and are not placed in the LLM prompt. The selected LLM provider still receives the prompts, retrieved context, and results needed to answer the user’s request.
- Dataset schema retrieval context is held in process memory and rebuilt from the saved dataset when needed. It is not a persistent cloud vector database. ChromaDB’s default embedding model may need to be downloaded on first indexing, so first use needs outbound network access.
- The sign-up form is open to visitors who can reach the app. For invite-only access, implement and verify an invitation workflow before disabling public sign-ups in Supabase.

## Requirements

- Python **3.12** is the recommended version for local development and Streamlit Community Cloud deployment. The app was developed with Python 3.12.
- A Supabase project with email/password Auth enabled.
- An API key from Gemini, OpenAI, or Anthropic for Analysis Chat. Data & Cleaning and EDA do not require an LLM key.
- The pinned Python dependencies in [`requirements.txt`](requirements.txt).

## Run locally

### 1. Clone the repository and create an environment

```bash
git clone <your-github-repository-url>
cd auto_analysis_RAG
python -m venv venv
```

Activate it, then install dependencies:

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# macOS or Linux
source venv/bin/activate
python -m pip install -r requirements.txt
```

Run Streamlit using the activated environment:

```bash
python -m streamlit run app.py
```

On Windows, if you prefer not to activate the environment first:

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

The app opens at `http://localhost:8501` by default.

### 2. Configure Supabase

1. Create a Supabase project.
2. In Supabase Auth settings, enable email/password sign-ups. Choose whether email confirmation is required. Set the local Site URL to `http://localhost:8501` while developing and allow that URL in the redirect URL list. See Supabase’s [Auth URL configuration](https://supabase.com/docs/guides/auth/redirect-urls), [row-level security](https://supabase.com/docs/guides/database/postgres/row-level-security), and [Storage access-control](https://supabase.com/docs/guides/storage/security/access-control) docs.
3. Open the project SQL Editor and run [`supabase_setup.sql`](supabase_setup.sql). This creates or configures the private `user-datasets` bucket with a 50 MiB object limit, creates the dataset metadata table, enables row-level security, and installs owner-scoped table and storage policies.
4. Copy [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) to `.streamlit/secrets.toml`, then fill in your project URL and publishable key:

   ```toml
   SUPABASE_URL = "https://<project-ref>.supabase.co"
   SUPABASE_PUBLISHABLE_KEY = "sb_publishable_..."
   ```

   A legacy Supabase anon key can instead be set as `SUPABASE_ANON_KEY`. These are the public project key and URL; **never place the Supabase service-role key in this app**.
5. Start the app, create an account, and sign in. Confirm the email first if confirmation is enabled.

### 3. Configure an LLM provider

After signing in, use the sidebar to select a provider, enter your own API key, and choose one of the models listed for that provider. The current model choices are configured in `app.py`; provider model availability can change over time. Your key is only required for Analysis Chat.

## Publish the project on GitHub

1. Create an empty GitHub repository. Do not initialize it with a README, license, or `.gitignore`; this project already contains those files.
2. From the project directory, initialize Git if needed, then stage and inspect the files before committing:

   ```bash
   git init
   git add .
   git status --short
   git diff --cached --name-only
   ```

3. Confirm that `.env`, `.streamlit/secrets.toml`, virtual environments, local workspace data, Python caches, and credentials are not staged. The repository’s `.gitignore` excludes these. Commit and push the source files, including `requirements.txt`, `.streamlit/config.toml`, `supabase_setup.sql`, and `kb/stats_guide.md`.
4. Connect and push the local branch to the GitHub repository from step 1 (replace the URL with that repository’s HTTPS or SSH URL):

   ```bash
   git commit -m "Prepare Auto-Analysis-RAG for deployment"
   git branch -M main
   git remote add origin <your-github-repository-url>
   git push -u origin main
   ```

5. Connect the GitHub repository to Streamlit Community Cloud. Select the branch to deploy and set the app entrypoint to `app.py`.

Do not commit real Supabase credentials, LLM API keys, uploaded datasets, or user exports. The example secrets file contains placeholders only.

## Deploy to Streamlit Community Cloud

1. Push the reviewed code to GitHub.
2. In Streamlit Community Cloud, create an app from that repository, branch, and the root `app.py` entrypoint.
3. In the deployment’s Advanced settings, select **Python 3.12** and add the following to the **Secrets** field:

   ```toml
   SUPABASE_URL = "https://<project-ref>.supabase.co"
   SUPABASE_PUBLISHABLE_KEY = "sb_publishable_..."
   ```

4. Deploy and copy the resulting app URL, for example `https://<your-app-name>.streamlit.app`.
5. In Supabase Auth URL configuration, change the **Site URL** to the deployed app URL and add that exact URL to the allowed redirect URLs. Keep `http://localhost:8501` in the allow list only if you still use the local app.
6. Open the deployed app, create or sign into an account, upload a small test dataset, and verify that it appears in Data & Cleaning and EDA. Then enter an LLM API key and make a simple descriptive Analysis Chat request. For an isolation check, create a second test account and verify it cannot see the first account’s dataset.

Streamlit Community Cloud installs the packages declared in `requirements.txt`. The repository includes `.streamlit/config.toml` to keep the upload limit at 50 MB. See the [Streamlit deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [dependency guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies), and [secrets/Python setup](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy#optional-configure-secrets-and-python-version) for current platform steps.

## Project layout

```text
auto_analysis_RAG/
├── .streamlit/
│   ├── config.toml                 # Streamlit upload-size configuration
│   └── secrets.toml.example        # Safe placeholder template; copy locally and fill in
├── app.py                          # Streamlit UI, authentication gate, and tabs
├── data_tools.py                   # Upload parsing, cleaning, profiling, EDA, charts
├── graph.py                        # LLM planning and deterministic tool orchestration
├── stats_ml_tools.py               # Statistical tests, supervised ML, forecasting
├── rag.py                           # In-memory retrieval context
├── supabase_backend.py             # Supabase Auth, Storage, and REST helpers
├── supabase_setup.sql              # Private bucket, metadata table, and RLS policies
├── kb/stats_guide.md               # Statistical and modeling reference
├── requirements.txt                # Pinned Python dependencies
├── LICENSE
└── README.md
```

## Troubleshooting

- **“Supabase is not configured”**: confirm `SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY` are present in local `.streamlit/secrets.toml` or Streamlit Community Cloud Secrets, then restart/reboot the app.
- **Sign-up or sign-in fails**: check Supabase email/password settings, whether email confirmation is required, and Supabase Auth logs. If confirmation is enabled, confirm the address before signing in.
- **Upload/storage permission error**: verify that `supabase_setup.sql` ran in the same Supabase project as the app’s secrets. Confirm the bucket is private and named `user-datasets`, and that the Storage and table policies are present.
- **Analysis module fails to initialize**: run `python -m pip install -r requirements.txt` in the same Python environment used to launch Streamlit, then start with `python -m streamlit run app.py`.
- **LLM request fails**: verify the selected provider’s API key, account access, and the selected model name. Provider errors and model availability are controlled by the provider.
- **Upload is rejected**: each source upload must be 50 MiB or smaller and contain no more than 200,000 rows. The generated Parquet copy and cleaned file must also each fit within the storage bucket’s 50 MiB per-object limit.

## License

This project is licensed under the Apache License 2.0. See [`LICENSE`](LICENSE).
