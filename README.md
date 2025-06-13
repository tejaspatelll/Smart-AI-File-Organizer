# 🪄 Smart File Organizer

> Transform chaos into order with AI-powered file organization that understands your intent.

**Smart File Organizer** is a sophisticated desktop application that leverages artificial intelligence to intelligently organize your files based on content analysis, custom prompts, and proven organizational templates. Built with an Apple-inspired interface that prioritizes user experience and accessibility.

## ✨ Key Features

### 🧠 **Intelligent AI Organization**
- **Content Analysis**: AI understands file contents, not just extensions
- **Custom Prompts**: Describe exactly how you want files organized in natural language
- **Smart Templates**: Choose from professionally-designed organization patterns
- **Context Awareness**: AI considers file relationships and usage patterns
- **Confidence Scoring**: Know how certain the AI is about each organizational decision (higher priority for files needing moving)

### 🎨 **Apple-Inspired Design**
- **Progressive Disclosure**: Start simple, reveal complexity as needed
- **Smooth Animations**: Fluid transitions and micro-interactions
- **Glassmorphism Effects**: Modern, depth-aware UI elements with consistent design system using CSS custom properties
- **Accessibility First**: Full keyboard navigation and screen reader support
- **Responsive Design**: Adapts beautifully to different screen sizes

### 🚀 **Advanced Capabilities**
- **Duplicate Detection**: Intelligent duplicate identification with smart best-file selection
- **Batch Processing**: Handle thousands of files efficiently, reducing API calls
- **Undo System**: Safely reverse any organizational changes
- **Priority-based Organization**: Focus on the most important files first, based on AI confidence
- **Real-time Progress**: Beautiful animated progress indicators with detailed status during scanning and analyzing
- **Preserves Folder Structure**: Intelligent logic to preserve existing folder structures and move files only when necessary

### 🎯 **Organization Methods**

#### **Quick & Smart** ⚡
Let AI automatically analyze your files and create the perfect organization structure with zero configuration.

#### **Custom Prompts** 💬
Describe your organizational vision in natural language:
- *"Organize by project type, then by date. Put all invoices in a separate folder."*
- *"Group photos by events and year. Keep work documents separate from personal files."*
- *"Create folders for each client, with subfolders for contracts, invoices, and communication."*

#### **Professional Templates** 📋
Choose from proven organizational patterns:

- **🎨 Creative Professional**: Project-based structure with client separation and asset libraries
- **💼 Business Professional**: Department-based organization with importance hierarchy
- **🎓 Student**: Subject and academic year organization with research materials
- **🏠 Personal Life**: Life category organization with event-based photo sorting

#### **Advanced Configuration** ⚙️
Fine-tune every aspect:
- File type inclusion/exclusion
- Confidence thresholds
- Priority settings
- Archive creation rules
- Smart naming conventions

## 🎬 **User Experience Flow**

### 1. **Welcome & Onboarding**
Beautiful welcome screen with three quick-start options that guide users to their preferred organizational method.

### 2. **Folder Selection**
Drag-and-drop or browse interface with visual feedback and folder validation.

### 3. **Method Selection**
Tabbed interface allowing users to choose and configure their preferred organization approach with real-time validation.

### 4. **Processing**
Animated progress indicators showing scanning and AI analysis phases with detailed statistics.

### 5. **Review & Customize**
Comprehensive preview with filtering, sorting, and selective application of organizational changes, featuring Table, Cards, and Grouped views, real-time search, filter controls, Quick Actions dropdown, and a compact, information-dense layout with inline AI reasoning and clear "FROM → TO" directory paths. Interface popups appear centered on the screen.

### 6. **Success & Follow-up**
Celebration of completion with statistics and easy access to organized folder, undo options, and new organization sessions.

## 🚀 **Quick Start**

### Prerequisites
- Python 3.8+ with pip
- Node.js 14+ (preferably 16+) with npm
- Google Gemini API key (for AI features)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/tejaspatelll/smart-file-organizer.git
   cd smart-file-organizer
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Node.js dependencies**
   ```bash
   npm install
   ```

4. **Set up AI integration**
   ```bash
   export GEMINI_API_KEY=your_api_key_here
   # Or create a .env file with GEMINI_API_KEY=your_api_key_here
   ```

5. **Launch the application**
   ```bash
   npm start
   ```

## 🔧 **CLI Usage**

For power users and automation, Smart File Organizer includes a sophisticated command-line interface:

### Basic Organization
```bash
python -m backend.cli --path /path/to/folder
```

### Intelligent AI Organization (Recommended)
```bash
python -m backend.cli --path /path/to/folder --intelligent
```

### Custom Prompts
```bash
python -m backend.cli --path /path/to/folder --prompt "Organize by project and date, separate work from personal" --template "Creative, Business, Student, Personal"
```

### Templates
```bash
python -m backend.cli --path /path/to/folder --template creative
```

### Advanced Options
```bash
python -m backend.cli --path /path/to/folder --include-duplicates --summary
```

### Apply Specific Changes
```bash
python -m backend.cli --path /path/to/folder --apply < plan.json
```

### Undo Recent Changes
```bash
python -m backend.cli --path /path/to/folder --undo --limit 25
```

## 📊 **Supported File Types**

- **Documents**: PDF, DOCX, DOC, TXT, MD, PPT, XLS, ODT
- **Images**: PNG, JPG, JPEG, WEBP, HEIC, GIF, BMP, TIFF, SVG
- **Videos**: MP4, AVI, MKV, MOV, WMV, FLV, WEBM, M4V
- **Audio**: MP3, WAV, FLAC, AAC, OGG, WMA, M4A
- **Code**: PY, JS, HTML, CSS, JSON, XML, YAML, YML
- **Archives**: ZIP, RAR, 7Z, TAR, GZ, BZ2
- **Applications**: EXE, MSI, DMG, PKG, DEB, RPM

## 🧪 **Example Custom Prompts**

### Creative Workflow
```
"Organize by project type: Design, Photography, Video. 
Within each type, create folders for each client. 
Keep work-in-progress separate from completed projects. 
Archive projects older than 1 year."
```

### Business Organization
```
"Separate by department: Finance, HR, Marketing, Operations. 
Within Finance, organize by document type and year. 
Keep important contracts in an easy-access folder. 
Archive documents older than 3 years."
```

### Academic Research
```
"Group by research topic and academic year. 
Separate published papers from working drafts. 
Create a references folder for each major project. 
Keep course materials organized by semester."
```

### Personal Life Management
```
"Organize photos by events and year. 
Group important documents: Financial, Medical, Legal. 
Keep travel documents together. 
Separate family photos by person and event."
```

## 🎨 **Design Principles**

### **Progressive Disclosure**
Start with simple options and reveal complexity only when needed. New users can organize files with one click, while power users can access advanced configuration.

### **Immediate Feedback**
Every interaction provides instant visual feedback through animations, progress indicators, and state changes.

### **Forgiveness**
Comprehensive undo system ensures users can explore organization options without fear of losing their current file structure.

### **Accessibility**
- Full keyboard navigation support
- Screen reader compatibility
- High contrast mode support
- Reduced motion options for sensitive users
- Clear focus indicators throughout

### **Performance**
- Efficient batch processing for large file collections
- Streaming progress updates during long operations
- Intelligent file type filtering to avoid processing system files
- Memory-efficient duplicate detection using hash algorithms

## 🔧 **Architecture**

### **Backend (Python)**
- **DirectoryScanner**: Recursive file discovery with intelligent filtering
- **CustomPromptClassifier**: Advanced AI integration with prompt-based organization, enabling understanding of user prompts
- **GeminiClassifier**: Content-aware file classification using Google's Gemini AI
- **FileOrganizer**: Smart folder matching and organization plan generation
- **CLI Interface**: Comprehensive command-line tool for automation

### **Frontend (Electron + HTML/CSS/JS)**
- **Apple-inspired UI**: Modern glassmorphism design with smooth animations, consistent design system with CSS custom properties, and beautiful UI
- **Progressive Web App principles**: Offline-capable with responsive design
- **Real-time Communication**: IPC-based communication with Python backend
- **State Management**: Sophisticated client-side state management for complex workflows

### **AI Integration**
- **Content Analysis**: Deep understanding of file contents, not just extensions
- **Pattern Recognition**: Learning from existing folder structures
- **Intent Understanding**: Natural language processing for custom prompts
- **Confidence Scoring**: Transparent AI decision-making with confidence levels

## 📈 **Performance Metrics**

- **File Processing**: 1000+ files per minute
- **AI Classification**: 25 files per API batch (optimized for accuracy and speed)
- **Memory Usage**: < 100MB for typical 10,000 file organization
- **Duplicate Detection**: MD5-based hashing with intelligent best-file selection
- **UI Responsiveness**: < 100ms interaction feedback, 60fps animations

## 🛠️ **Development**

### **Project Structure**
```
smart-file-organizer/
├── backend/              # Python backend
│   ├── __init__.py
│   ├── cli.py           # Command-line interface
│   ├── organizer.py     # Core organization logic
│   └── utils.py         # Utility functions
├── frontend/            # Electron frontend
│   ├── main.js          # Electron main process
│   ├── preload.js       # IPC bridge
│   └── renderer/        # UI components
│       ├── index.html
│       ├── style.css
│       └── renderer.js
├── requirements.txt     # Python dependencies
├── package.json        # Node.js dependencies
└── README.md
```

### **Contributing**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### **Running Tests**
```bash
# Python tests
python -m pytest tests/

# Frontend tests
npm test
```

### **Development Mode**
```bash
# Start with hot reload
npm run dev
```

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Google Gemini AI** for advanced content understanding
- **Bootstrap 5** for responsive UI components
- **Bootstrap Icons** for beautiful iconography
- **Electron** for cross-platform desktop application framework
- **Apple Human Interface Guidelines** for design inspiration

## 📞 **Support**

- 📖 **Documentation**: [Full documentation available](https://github.com/yourusername/smart-file-organizer/wiki)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/yourusername/smart-file-organizer/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/yourusername/smart-file-organizer/discussions)
- 💬 **Community**: [Discord Server](https://discord.gg/yourserver)

---

**Transform your file chaos into organized bliss with Smart File Organizer. Because life's too short for messy folders.** ✨ 

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 🐛 Troubleshooting

### Common Issues

#### "Organization Failed" Error
- **Check API Key**: Ensure `GEMINI_API_KEY` is set correctly
- **Verify Permissions**: Make sure you have read/write access to the folder
- **Check Internet**: AI features require an internet connection
- **Try Smaller Folder**: Very large folders (10k+ files) may timeout

#### Application Won't Start
- **Python Version**: Ensure Python 3.8+ is installed
- **Dependencies**: Run `pip install -r requirements.txt` in activated venv
- **Node Version**: Ensure Node.js 16+ is installed
- **Clear Cache**: Delete `node_modules` and run `npm install` again

#### No Files Being Organized
- **Hidden Files**: The app skips system and hidden files by default
- **File Types**: Some file types may not be supported for AI analysis
- **Empty Folders**: Folders with only system files will show as empty
- **Already Organized**: Well-organized folders may not need changes

#### Performance Issues
- **Large Files**: Files >100MB are skipped by default for performance
- **Network Speed**: AI processing requires stable internet connection
- **System Resources**: Close other applications if running slowly
- **Batch Size**: Reduce batch size for slower systems
- **GUI Glitches/Tile Memory Exceeded**: This can occur due to complex visual effects in the UI. Ensure `will-change` CSS property is applied to animated or transformed elements to optimize rendering. Restarting the application after applying changes is recommended.

### Getting Help

1. **Check the logs**: Look in the developer console for error details
2. **Try safe mode**: Use Quick & Smart mode without custom prompts
3. **Test with small folder**: Verify functionality with a small test folder
4. **Check system requirements**: Ensure all prerequisites are met

### Reporting Issues

When reporting issues, please include:
- Operating system and version
- Python and Node.js versions
- Error messages (full text)
- Steps to reproduce the issue
- Example folder structure (if relevant)

---

Made with ❤️ for organized minds everywhere. 
