# Contributing to AstroVision 🚀

We love your input! We want to make contributing to AstroVision as easy and transparent as possible.

## 🎯 **Areas for Contribution**

### 🌌 **Model Improvements**
- [ ] New astronomical object classes (quasars, pulsars, supernovas)
- [ ] Advanced architectures (Vision Transformers, EfficientNet)
- [ ] Model compression and optimization
- [ ] Multi-modal learning (spectral + visual data)

### 🔬 **Feature Engineering**
- [ ] Advanced astronomical feature extraction
- [ ] FITS format processing improvements
- [ ] Spectral analysis integration
- [ ] Time-series analysis for variable stars

### 🚀 **Infrastructure**
- [ ] REST API development
- [ ] Mobile app (React Native/Flutter)
- [ ] Web interface enhancement
- [ ] Cloud deployment automation

### 📊 **Data & Research**
- [ ] Dataset expansion (NASA, ESA archives)
- [ ] Synthetic data generation
- [ ] Research paper reproduction
- [ ] Benchmark comparisons

## 🛠️ **Development Setup**

### **1. Fork & Clone**
```bash
git clone https://github.com/yourusername/astrovision.git
cd astrovision
```

### **2. Environment Setup**
```bash
# Create development environment
python -m venv dev-env
source dev-env/bin/activate  # Linux/Mac
# dev-env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### **3. Pre-commit Hooks**
```bash
pre-commit install
```

## 📋 **Development Guidelines**

### **Code Style**
- Use `black` for code formatting
- Follow PEP 8 guidelines
- Add type hints where possible
- Write descriptive variable names

### **Testing**
```bash
# Run tests
python -m pytest tests/

# Run with coverage
pytest --cov=scripts tests/
```

### **Documentation**
- Update README for new features
- Add docstrings to all functions
- Include usage examples

## 🚀 **Pull Request Process**

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/amazing-new-feature
   ```

2. **Make Changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation

3. **Test Everything**
   ```bash
   # Format code
   black scripts/ notebooks/
   
   # Run tests
   pytest tests/
   
   # Check model performance
   python scripts/test_improved_model.py
   ```

4. **Commit with Conventional Commits**
   ```bash
   git commit -m "feat: add quasar classification support"
   git commit -m "fix: resolve FITS file loading issue"
   git commit -m "docs: update API documentation"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/amazing-new-feature
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

- 💬 Open a [Discussion](https://github.com/yourusername/astrovision/discussions)
- 🐛 Report a [Bug](https://github.com/yourusername/astrovision/issues)
- 💡 Request a [Feature](https://github.com/yourusername/astrovision/issues)

---

**Thank you for contributing to AstroVision! 🌌**
