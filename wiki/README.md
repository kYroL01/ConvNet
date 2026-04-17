# Wiki Content for GitHub Wiki

This directory contains comprehensive documentation for the ConvNet project. These markdown files are ready to be uploaded to the GitHub Wiki.

## Wiki Pages

1. **Home.md** - Wiki homepage with table of contents
2. **AlexNet-Architecture-Theory.md** - Deep dive into the model architecture and theory
3. **Advanced-Training-Configuration.md** - Detailed guide on all training parameters
4. **Data-Preparation-Guide.md** - Complete guide to preparing your dataset
5. **Hyperparameter-Tuning.md** - Strategies for optimizing model performance
6. **Model-Performance-and-Metrics.md** - Understanding evaluation metrics
7. **Advanced-Troubleshooting.md** - Solutions to common and advanced issues
8. **TensorFlow-2x-Migration-Notes.md** - Migration details from TensorFlow 1.x

## How to Upload to GitHub Wiki

## Implemented Operations

The operations described in this README are implemented in:

```bash
python wiki/wiki_operations.py [--wiki-dir <path>] <command> [options]
```

### 1) Copy Wiki Pages to a Local `.wiki` Repository (Method 2 helper)

```bash
# Example: assumes ConvNet.wiki already exists locally
python wiki/wiki_operations.py copy --dest /path/to/ConvNet.wiki
```

Dry run:

```bash
python wiki/wiki_operations.py copy --dest /path/to/ConvNet.wiki --dry-run
```

### 2) Create Wiki Pages with GitHub CLI (Method 3)

```bash
python wiki/wiki_operations.py gh-create -R kYroL01/ConvNet
```

Dry run:

```bash
python wiki/wiki_operations.py gh-create -R kYroL01/ConvNet --dry-run
```

### 3) Verify Required Wiki Pages and Internal Links

```bash
python wiki/wiki_operations.py verify
```

### Method 1: Using GitHub Web Interface

1. Go to your repository on GitHub: https://github.com/kYroL01/ConvNet
2. Click on the "Wiki" tab at the top
3. If Wiki is not enabled, click "Create the first page" to enable it
4. For each file:
   - Click "New Page"
   - Copy the filename (without .md extension) as the page title
   - Copy and paste the content from the file
   - Click "Save Page"

### Method 2: Using Git (Recommended)

The GitHub Wiki is actually a separate Git repository. You can clone and push to it:

```bash
# Clone the wiki repository
git clone https://github.com/kYroL01/ConvNet.wiki.git

# Copy wiki files
cp wiki/*.md ConvNet.wiki/

# Commit and push
cd ConvNet.wiki
git add .
git commit -m "Add comprehensive Wiki documentation"
git push origin master
```

### Method 3: Using GitHub CLI

```bash
# Ensure you have gh CLI installed and authenticated
gh auth login

# Navigate to wiki directory
cd wiki

# Create each page
gh wiki create Home -F Home.md
gh wiki create AlexNet-Architecture-Theory -F AlexNet-Architecture-Theory.md
gh wiki create Advanced-Training-Configuration -F Advanced-Training-Configuration.md
gh wiki create Data-Preparation-Guide -F Data-Preparation-Guide.md
gh wiki create Hyperparameter-Tuning -F Hyperparameter-Tuning.md
gh wiki create Model-Performance-and-Metrics -F Model-Performance-and-Metrics.md
gh wiki create Advanced-Troubleshooting -F Advanced-Troubleshooting.md
gh wiki create TensorFlow-2x-Migration-Notes -F TensorFlow-2x-Migration-Notes.md
```

## Page Naming Convention

When creating wiki pages on GitHub:
- Use the filename without the `.md` extension as the page title
- GitHub will automatically convert hyphens to spaces in the display
- Internal links use the hyphenated format

## Wiki Structure

```
Home (landing page)
├── AlexNet Architecture Theory
├── Advanced Training Configuration
├── Data Preparation Guide
├── Hyperparameter Tuning
├── Model Performance and Metrics
├── Advanced Troubleshooting
└── TensorFlow 2x Migration Notes
```

## Content Summary

### What's Covered

- **Theory**: AlexNet architecture, CNNs, deep learning concepts
- **Practical**: All command-line options, parameters, configurations
- **Advanced**: Hyperparameter tuning, performance optimization
- **Troubleshooting**: Common issues, debugging, error messages
- **Migration**: TensorFlow 1.x to 2.x migration details

### README vs. Wiki

As per the issue requirements:
- **README**: Covers the basics (installation, quick start, basic usage)
- **Wiki**: Explores topics in depth with detailed explanations and theory

## Verification

After uploading to GitHub Wiki:

1. Check all pages are created
2. Verify internal links work
3. Ensure images (if any) are displayed
4. Test navigation between pages
5. Confirm external links (to README, etc.) work

## Maintenance

To update wiki pages:
1. Edit the markdown files in this directory
2. Re-upload using one of the methods above
3. Keep this directory in sync with GitHub Wiki

## Notes

- These wiki files are included in the repository for version control
- The actual GitHub Wiki is a separate git repository
- Keep both in sync when making updates
- Wiki pages support GitHub Flavored Markdown
