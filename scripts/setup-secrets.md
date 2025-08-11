# GitHub Secrets Setup Guide

To enable automated Docker Hub deployment, you need to configure the following GitHub secrets in your repository settings.

## Required Secrets

### 1. DOCKER_USERNAME
- **Description**: Your Docker Hub username
- **Value**: `gemneye` (or your Docker Hub username)
- **Usage**: Used for Docker Hub authentication during CI/CD

### 2. DOCKER_PASSWORD
- **Description**: Docker Hub access token (recommended) or password
- **Value**: Your Docker Hub access token or password
- **Usage**: Used for Docker Hub authentication during CI/CD

## Setting up Secrets

1. **Go to your GitHub repository**:
   - Navigate to: https://github.com/sruckh/CharaConsist-runpod

2. **Access Settings**:
   - Click on the "Settings" tab
   - In the left sidebar, click "Secrets and variables"
   - Click "Actions"

3. **Add Repository Secrets**:
   - Click "New repository secret"
   - Name: `DOCKER_USERNAME`
   - Secret: `gemneye`
   - Click "Add secret"

4. **Add Docker Password/Token**:
   - Click "New repository secret" again
   - Name: `DOCKER_PASSWORD`  
   - Secret: `[Your Docker Hub access token or password]`
   - Click "Add secret"

## Creating Docker Hub Access Token (Recommended)

Instead of using your Docker Hub password, create an access token:

1. **Log in to Docker Hub**:
   - Go to https://hub.docker.com/
   - Sign in to your account

2. **Generate Access Token**:
   - Go to Account Settings → Security
   - Click "New Access Token"
   - Name: `GitHub-Actions-CharaConsist`
   - Permissions: `Read, Write, Delete`
   - Click "Generate"
   - Copy the token (you won't see it again)

3. **Use Token as Secret**:
   - Use this token as the value for `DOCKER_PASSWORD` secret

## Verification

Once secrets are configured:

1. **Push to main branch** or create a **pull request**
2. **Check Actions tab** in your GitHub repository
3. **Monitor the workflow** execution
4. **Verify deployment** on Docker Hub: https://hub.docker.com/r/gemneye/characonsist

## Security Notes

- Never commit Docker Hub credentials to your repository
- Use access tokens instead of passwords when possible
- Regularly rotate access tokens for security
- Monitor access token usage in Docker Hub dashboard

## Troubleshooting

### Authentication Failed
- Verify `DOCKER_USERNAME` matches your Docker Hub username
- Ensure `DOCKER_PASSWORD` contains valid access token/password
- Check if access token has proper permissions

### Build Failed
- Check workflow logs in GitHub Actions
- Verify Dockerfile syntax and dependencies
- Ensure all required files are included in the repository

### Push Failed
- Confirm repository exists on Docker Hub: `gemneye/characonsist`
- Verify access token has `Write` permissions
- Check Docker Hub rate limits and storage quotas