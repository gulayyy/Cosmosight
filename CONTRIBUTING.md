# Contributing to Astronomical Classification Project 🚀

We welcome contributions to improve our Galaxy/Nebula/Star classification system!

## 🎯 **Current Project Scope**

This project uses **MobileNetV2** to classify astronomical images with **99.47% test accuracy**.

### � **Areas for Contribution**

### **Model & Performance**
- [ ] Model architecture improvements (keeping MobileNetV2 base)
- [ ] Training optimization and hyperparameter tuning
- [ ] Data augmentation strategies
- [ ] Model evaluation and testing improvements

### **Code Quality**
- [ ] Code documentation and comments
- [ ] Error handling improvements
- [ ] Code refactoring and optimization
- [ ] Cross-platform compatibility testing

### **Data & Analysis**
- [ ] Dataset expansion (more Galaxy/Nebula/Star images)
- [ ] Image preprocessing improvements
- [ ] Analysis visualization enhancements
- [ ] Performance metrics and reporting

### **Usability**
- [ ] User interface improvements
- [ ] Command-line argument handling
- [ ] Better prediction output formatting
- [ ] Installation and setup documentation

## 🛠️ **Development Setup**

### **1. Clone Repository**
```bash
git clone <your-repository-url>
cd astro_classification_project
```

### **2. Environment Setup**
```bash
# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Ubuntu/Linux 
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### **3. Verify Installation**
```bash
# Test model loading
python scripts/improved_predict.py data/processed_images/Galaxy/100.jpg

# Check training setup
python scripts/train_model.py --help
```

## 📋 **Development Guidelines**

### **Code Style**
- Use clear, descriptive variable names
- Add comments for complex logic
- Follow Python PEP 8 guidelines
- Write helpful docstrings

### **File Organization**
- Keep scripts in `scripts/` directory
- Save models in `models/` directory
- Store datasets in `data/` directory
- Document changes in code comments

### **Testing Your Changes**
```bash
# Test prediction functionality
python scripts/improved_predict.py data/processed_images/Galaxy/100.jpg

# Test training pipeline
python scripts/train_model.py

# Check utility functions
python scripts/utils.py
```

### **Documentation**
- Update README for new features
- Add docstrings to all functions
- Include usage examples

## 🚀 **Pull Request Process**

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write clean, documented code
   - Test your changes thoroughly
   - Update documentation if needed

3. **Test Everything**
   ```bash
   # Test prediction functionality
   python scripts/improved_predict.py data/processed_images/Galaxy/100.jpg
   
   # Test model training
   python scripts/train_model.py
   
   # Check utilities
   python scripts/utils.py
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "Add: descriptive commit message"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## 🏷️ **Issue Labels**

- `bug` - Something isn't working
- `enhancement` - New feature or request  
- `documentation` - Improvements to docs
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention is needed
- `question` - Further information is requested

## 🎉 **Recognition**

Contributors will be:
- ✅ Added to Contributors section
- ✅ Mentioned in release notes
- ✅ Given credit in research papers
- ✅ Invited to project discussions

## 📞 **Questions?**

- 💬 Open a Discussion on GitHub
- 🐛 Report a Bug via Issues
- 💡 Request a Feature via Issues

---

**Thank you for contributing to Astronomical Classification Project! 🌌**
