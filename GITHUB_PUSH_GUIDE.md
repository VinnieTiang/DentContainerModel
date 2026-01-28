# GitHub Push Guide - Fix HTTP 400 Error

## Problem

You're getting `HTTP 400` error when pushing. This is usually an **authentication issue**.

## Solution Options

### Option 1: Use Personal Access Token (Recommended for HTTPS)

GitHub no longer accepts passwords for HTTPS. You need a **Personal Access Token**:

1. **Create a Token:**

   - Go to: https://github.com/settings/tokens
   - Click "Generate new token" → "Generate new token (classic)"
   - Name it: `DentContainerModel-push`
   - Select scopes: Check `repo` (full control of private repositories)
   - Click "Generate token"
   - **COPY THE TOKEN** (you won't see it again!)

2. **Use the Token:**

   ```bash
   git push --set-upstream origin main
   ```

   - Username: Your GitHub username (`VinnieTiang`)
   - Password: **Paste your Personal Access Token** (not your GitHub password)

3. **Save Credentials (Optional):**
   ```bash
   git config --global credential.helper osxkeychain  # macOS
   ```
   This saves your token so you don't need to enter it every time.

### Option 2: Use SSH (More Secure)

1. **Check if you have SSH keys:**

   ```bash
   ls -la ~/.ssh/id_*.pub
   ```

2. **If no keys exist, generate one:**

   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   # Press Enter to accept default location
   # Optionally set a passphrase
   ```

3. **Add SSH key to GitHub:**

   ```bash
   cat ~/.ssh/id_ed25519.pub
   # Copy the output
   ```

   - Go to: https://github.com/settings/keys
   - Click "New SSH key"
   - Paste your public key
   - Save

4. **Switch to SSH:**
   ```bash
   git remote set-url origin git@github.com:VinnieTiang/DentContainerModel.git
   git push --set-upstream origin main
   ```

### Option 3: Use GitHub CLI (Easiest)

1. **Install GitHub CLI:**

   ```bash
   brew install gh  # macOS
   ```

2. **Authenticate:**

   ```bash
   gh auth login
   ```

3. **Push:**
   ```bash
   git push --set-upstream origin main
   ```

## Quick Fix Commands

Try these in order:

```bash
# 1. Increase HTTP buffer (already done)
git config http.postBuffer 524288000

# 2. Try pushing again
git push --set-upstream origin main

# If still fails, use Personal Access Token (Option 1 above)
```

## Verify Repository Exists

Make sure the repository exists:

- Check: https://github.com/VinnieTiang/DentContainerModel
- If it doesn't exist, create it on GitHub first

## Troubleshooting

**"Everything up-to-date" but push fails:**

- This might mean the push partially succeeded
- Check your GitHub repository online
- If files are there, you're done!

**Still getting HTTP 400:**

- Make sure you're using a Personal Access Token, not password
- Check repository permissions
- Try creating a new repository and pushing to it
