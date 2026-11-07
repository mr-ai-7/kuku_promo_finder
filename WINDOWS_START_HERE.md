# Kuku Promo Finder Version 2: Windows + VS Code

## Choice A — Create a fresh environment inside the Version 2 folder

1. Download the new ZIP, right-click → **Extract All**.
2. Open VS Code → **File → Open Folder** → select **Kuku_Promo_Finder_V2** (the folder containing `prototype`).
3. In VS Code, choose **Terminal → New Terminal** (PowerShell).
4. Run these commands **one by one**:

```powershell
Get-Location
dir
py --version
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\prototype\requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s .\testing -v
.\.venv\Scripts\python.exe -m streamlit run .\prototype\app.py
```

Open http://localhost:8501 . If Version 1 is still running on port 8501, stop it using `Ctrl + C` in its terminal **or** launch v2 using:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\prototype\app.py --server.port 8502
```

Then open http://localhost:8502 .

## Choice B — Reuse your existing environment

Your previous `.venv` was inside `C:\Users\jatav\OneDrive\Desktop\kuku`. If you extract the v2 ZIP inside that directory, follow these steps **from the kuku parent directory**, leaving the existing `.venv` in place:

```powershell
cd "C:\Users\jatav\OneDrive\Desktop\kuku"
.\.venv\Scripts\python.exe -m pip install -r ".\Kuku_Promo_Finder_V2\prototype\requirements.txt"
.\.venv\Scripts\python.exe -m unittest discover -s ".\Kuku_Promo_Finder_V2\testing" -v
.\.venv\Scripts\python.exe -m streamlit run ".\Kuku_Promo_Finder_V2\prototype\app.py" --server.port 8502
```

Visit http://localhost:8502 .

**These commands assume the extracted folder is named `Kuku_Promo_Finder_V2`.** If Windows names it differently, check `dir` and use the actual name.

## Uploading and exporting

- **Upload to the app:** the original company scripts ZIP supplied for the assignment (do not upload this ZIP to GitHub).
- **Sidebar:** 20 ranked promos, 1 maximum per episode.
- **Analyze:** you should see 50 episode files for the supplied company pack. The number of candidate clips will differ from v1 because v2 uses different boundary and scoring rules.
- **Download:** the six raw output files. These belong in a private folder link for the hiring submission.
- **Demo:** first try `demo_input/demo_episode_001.txt`, then test the two-episode folder from the CLI.

## If something goes wrong

- `No module named streamlit` → run the `pip install -r ...` command above using the **same** `.venv\Scripts\python.exe`.
- `Could not open requirements file` → your terminal is in the wrong folder; run `Get-Location` and `dir`.
- `Port 8501 is already in use` → stop v1 or use `--server.port 8502`.
- `Cannot parse enough timed script content` → confirm scripts contain `Start: (00:10.00) | End: (00:14.00)` entries and a minimum of three dialogue events per episode.
- If a different ZIP or slider setting is chosen, click **Analyze** again. V2 hides stale results.

## Deployment reminder

GitHub repo: upload the `prototype`, `demo_input`, `testing` directories, root README, Windows guide and `.gitignore` only. Do not upload the original Kuku ZIP, its scripts, private raw results, secrets, or `.venv`.

Streamlit Cloud main file path: `prototype/app.py`.
