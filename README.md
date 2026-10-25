# Newsletter Generator with CrewAI

A Streamlit web application that generates newsletters on any topic using CrewAI. The application converts the generated content into a downloadable PDF format.

## Features

- 🤖 AI-powered newsletter generation using CrewAI
- 📝 Simple topic-based input interface
- 📄 Automatic PDF conversion
- 👀 Real-time PDF preview
- ⬇️ PDF download capability
- 🎯 User-friendly Streamlit interface

## Installation

1. Clone the repository:

```bash
git clone https://github.com/bnarasimha21/crewai-newsletter-generator.git
cd newsletter-generator
```

2. Create and activate a virtual environment:

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n newsletter-env python=3.11
conda activate newsletter-env
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Install wkhtmltopdf (required for PDF generation):

**For macOS:**
```bash
brew install wkhtmltopdf
```

**For Ubuntu/Debian:**
```bash
sudo apt-get install wkhtmltopdf
```

**For Windows:**
Download and install from: https://wkhtmltopdf.org/downloads.html

5. Set the API keys the crew needs (as environment variables):

```bash
export OPENAI_API_KEY=...   # default LLM used by the CrewAI agents
export SERPER_API_KEY=...   # web search tool used by the researcher agent
```

## Project Structure

```
newsletter-generator/
├── src/
│   ├── crew.py                  # researcher, reporting_analyst, pdf_generator agents and tasks
│   ├── config/
│   │   ├── agents.yaml
│   │   └── tasks.yaml
│   └── tools/
│       └── html_to_pdf_tool.py  # pdfkit/wkhtmltopdf tool
├── app.py
├── requirements.txt
└── README.md
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Enter a topic in the input field and click "Generate Newsletter"

4. Wait for the generation process to complete

5. View the generated PDF directly in the browser

6. Download the PDF using the "Download PDF" button

7. View this session's generation history in the table below (it resets when the page is reloaded)

## Requirements

- Python 3.10+
- CrewAI
- Streamlit
- pdfkit
- wkhtmltopdf
- pandas

See `requirements.txt` for complete list of dependencies.

## Configuration

The application uses the following configuration files:
- CrewAI agent and task configuration in `src/config/` (`agents.yaml`, `tasks.yaml`)
- PDF generation in `src/tools/html_to_pdf_tool.py`

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CrewAI for the AI generation capabilities
- Streamlit for the web interface
- wkhtmltopdf for PDF conversion

## Support

For support, please open an issue in the GitHub repository or contact [your-email@example.com].

## Roadmap

- [ ] Add support for custom templates
- [ ] Implement user authentication
- [ ] Add more customization options
- [ ] Enable sharing capabilities
- [ ] Add export options for different formats

## Authors

- Your Name (@yourgithubhandle)

## Version History

- 0.1.0
    - Initial Release
    - Basic newsletter generation
    - PDF conversion
    - History tracking
