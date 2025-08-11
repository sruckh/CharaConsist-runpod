# CI/CD Pipeline Setup Guide

This document describes the automated CI/CD pipeline for CharaConsist container deployment.

## Overview

The CI/CD pipeline automatically builds and deploys Docker containers to Docker Hub (`gemneye/characonsist`) when changes are pushed to the main branch.

## Pipeline Features

### 🚀 Automated Deployment
- **Trigger**: Push to `main` branch or manual workflow dispatch
- **Target**: Docker Hub repository `gemneye/characonsist`
- **Platform**: AMD64/x86_64 architecture only
- **Caching**: GitHub Actions cache for faster builds
- **Testing**: Automated container testing and validation

### 📋 Workflow Triggers

1. **Push to main branch**: Automatic build and push
2. **Pull requests**: Build only (no push) for testing
3. **Manual dispatch**: Manual workflow trigger with force rebuild option
4. **Path filtering**: Ignores documentation-only changes

### 🏷️ Image Tagging Strategy

- `latest` - Latest stable build from main branch
- `main-<sha>` - Main branch with short commit SHA
- `YYYY-MM-DD` - Date-based tags for main branch builds
- `pr-<number>` - Pull request builds (not pushed)

## Repository Structure

```
.github/
├── workflows/
│   └── docker-build-deploy.yml    # Main CI/CD pipeline
├── ISSUE_TEMPLATE/
│   ├── bug_report.md              # Bug report template
│   └── feature_request.md         # Feature request template
├── pull_request_template.md       # PR template
└── FUNDING.yml                    # Funding configuration

scripts/
├── docker-test.sh                 # Docker testing script
└── setup-secrets.md               # Secrets setup guide
```

## Setup Instructions

### 1. Configure GitHub Secrets

Set up the following secrets in your GitHub repository:

- `DOCKER_USERNAME`: Your Docker Hub username (`gemneye`)
- `DOCKER_PASSWORD`: Docker Hub access token or password

📖 **Detailed guide**: See [scripts/setup-secrets.md](../scripts/setup-secrets.md)

### 2. Verify Repository Settings

Ensure your repository has:
- Actions enabled
- Write permissions for GitHub Actions
- Proper branch protection rules (optional)

### 3. Test the Pipeline

1. Make changes to your code
2. Commit and push to main branch
3. Monitor the workflow in GitHub Actions tab
4. Verify deployment on Docker Hub

## Workflow Stages

### 1. Build Preparation
- Checkout repository code
- Set up Docker Buildx with AMD64 platform
- Extract metadata for tagging and labeling

### 2. Authentication & Build
- Log in to Docker Hub (main branch only)
- Build Docker image with caching
- Apply comprehensive labels and metadata

### 3. Testing & Validation
- Test container startup and Python environment
- Verify all dependencies are properly installed
- Check CUDA availability and inference script
- Run basic security scans

### 4. Deployment
- Push to Docker Hub (main branch builds only)
- Generate deployment summary
- Clean up build artifacts

## Performance Optimizations

### 🏎️ Build Optimizations
- **GitHub Actions Cache**: Speeds up subsequent builds
- **Docker Layer Caching**: Reuses unchanged layers
- **Selective Building**: Skip builds for documentation changes
- **Parallel Operations**: Concurrent testing and validation

### 📊 Build Metrics
- **Typical build time**: 15-25 minutes
- **Cache hit ratio**: 60-80% layer reuse
- **Image size**: ~8-12GB (with PyTorch + CUDA)
- **Platform support**: AMD64 only for optimal performance

## Usage Examples

### Pull Latest Image
```bash
docker pull gemneye/characonsist:latest
```

### Run Container
```bash
docker run -it --gpus all gemneye/characonsist:latest
```

### Run with Model Mounting
```bash
docker run -it --gpus all \
  -v /path/to/models:/workspace/models \
  -v /path/to/results:/workspace/characonsist/results \
  gemneye/characonsist:latest
```

### Test Container Locally
```bash
# Run the test script
./scripts/docker-test.sh gemneye/characonsist latest
```

## Monitoring & Troubleshooting

### 📊 Monitoring Locations
- **GitHub Actions**: Workflow execution and logs
- **Docker Hub**: Image repository and download metrics
- **Repository Insights**: Usage analytics and traffic

### 🔧 Common Issues

#### Build Failures
1. Check Dockerfile syntax
2. Verify all dependencies in requirements.txt
3. Ensure base image availability
4. Review build logs in GitHub Actions

#### Authentication Failures
1. Verify GitHub secrets are set correctly
2. Check Docker Hub token permissions
3. Ensure repository exists on Docker Hub
4. Validate access token expiration

#### Test Failures
1. Review test output in workflow logs
2. Check dependency compatibility
3. Verify file structure and permissions
4. Test container locally first

### 🚨 Emergency Procedures

#### Stop Automatic Deployments
1. Go to repository Settings → Actions
2. Disable Actions or disable specific workflow
3. Or temporarily rename the workflow file

#### Rollback Deployment
```bash
# Pull previous version
docker pull gemneye/characonsist:2024-08-10

# Or use specific commit SHA
docker pull gemneye/characonsist:main-abc123d
```

## Security Considerations

### 🔒 Security Features
- **Secret Management**: GitHub secrets for credentials
- **Access Tokens**: Recommended over passwords
- **Dependency Scanning**: Basic security checks
- **Container Security**: Regular base image updates

### 🛡️ Best Practices
- Regularly rotate access tokens
- Monitor Docker Hub for security alerts
- Keep base images updated
- Review and approve all pull requests
- Use branch protection rules

## Contributing

### For Pull Requests
- All PRs trigger build testing (no push)
- Include tests for new features
- Update documentation as needed
- Follow the PR template

### For Maintainers
- Review security implications
- Monitor build performance
- Update base images regularly
- Maintain secrets and access tokens

## Support

### Documentation
- [Docker Hub Repository](https://hub.docker.com/r/gemneye/characonsist)
- [GitHub Repository](https://github.com/sruckh/CharaConsist-runpod)
- [Issues and Bug Reports](https://github.com/sruckh/CharaConsist-runpod/issues)

### Getting Help
1. Check existing issues and documentation
2. Create detailed bug reports using issue templates
3. Include workflow logs and error messages
4. Test locally before reporting CI/CD issues