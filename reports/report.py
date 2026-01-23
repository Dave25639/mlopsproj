# run following commands to install requirements
# pip install typer markdown pydantic loguru
# or
# uv add typer markdown pydantic loguru

import re
from pathlib import Path

import markdown
import pydantic
import typer
from loguru import logger


class Constraints(pydantic.BaseModel):
    """Base class for constraints."""

    def __call__(self, answer: str, index: int) -> None:
        """Check constraints on the answer."""
        raise NotImplementedError


class NoConstraints(Constraints):
    """No constraints on the answer."""

    def __call__(self, answer: str, index: int) -> bool:
        """No constraints on the answer."""
        return True


class LengthConstraints(Constraints):
    """Check constraints on the length of the answer."""

    min_length: int = pydantic.Field(ge=0)
    max_length: int = pydantic.Field(ge=0)

    def __call__(self, answer: str, index: int) -> bool:
        """Check constraints on the length of the answer."""
        answer = answer.split()
        if not (self.min_length <= len(answer) <= self.max_length):
            logger.warning(
                f"Question {index} failed check. Expected number of words to be"
                f" between {self.min_length} and {self.max_length} but got {len(answer)}"
            )
            return False
        return True


class ImageConstraints(Constraints):
    """Check constraints on the number of images in the answer."""

    min_images: int = pydantic.Field(ge=0)
    max_images: int = pydantic.Field(ge=0)

    def __call__(self, answer: str, index: int) -> bool:
        """Check constraints on the number of images in the answer."""
        links = re.findall(r"\!\[.*?\]\(.*?\)", answer)
        if not (self.min_images <= len(links) <= self.max_images):
            logger.warning(
                f"Question {index} failed check. Expected number of screenshots to be"
                f" between {self.min_images} and {self.max_images} but got {len(links)}"
            )
            return False
        return True


class MultiConstraints(Constraints):
    """Check multiple constraints on the answer."""

    constrains: list[Constraints]

    def __call__(self, answer: str, index: int) -> None:
        """Check multiple constraints on the answer."""
        value = True
        for fn in self.constrains:
            value = fn(answer, index) and value
        return value


app = typer.Typer()


@app.command()
def html() -> None:
    """Convert README.md to html page."""
    path = "reports/README.md" if Path("reports/README.md").exists() else "README.md"
    with Path(path).open() as file:
        text = file.read()
    text = text[43:]  # remove header

    # Try to use extensions if available, otherwise use basic markdown
    try:
        html_content = markdown.markdown(text, extensions=['fenced_code', 'tables', 'nl2br'])
    except Exception:
        html_content = markdown.markdown(text)

    # CSS styling for a modern, professional look
    css = """
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }

        .container {
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
            font-size: 2.5em;
        }

        h2 {
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
            font-size: 2em;
        }

        h3 {
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 1.5em;
        }

        h4 {
            color: #555;
            margin-top: 20px;
            margin-bottom: 10px;
            font-size: 1.2em;
        }

        p {
            margin-bottom: 15px;
            text-align: justify;
        }

        ul, ol {
            margin-left: 30px;
            margin-bottom: 20px;
        }

        li {
            margin-bottom: 8px;
        }

        /* Checklist styling */
        ul li input[type="checkbox"] {
            margin-right: 8px;
            transform: scale(1.2);
        }

        blockquote {
            border-left: 4px solid #3498db;
            padding-left: 20px;
            margin: 20px 0;
            color: #555;
            font-style: italic;
            background-color: #f8f9fa;
            padding: 15px 20px;
            border-radius: 4px;
        }

        blockquote p {
            margin-bottom: 10px;
        }

        blockquote p:last-child {
            margin-bottom: 0;
        }

        code {
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            color: #e83e8c;
        }

        pre {
            background-color: #2d2d2d;
            color: #f8f8f2;
            padding: 20px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 20px 0;
        }

        pre code {
            background-color: transparent;
            color: inherit;
            padding: 0;
        }

        img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 20px 0;
            display: block;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }

        table th, table td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }

        table th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }

        table tr:nth-child(even) {
            background-color: #f2f2f2;
        }

        hr {
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 30px 0;
        }

        strong {
            color: #2c3e50;
            font-weight: 600;
        }

        em {
            color: #555;
        }

        a {
            color: #3498db;
            text-decoration: none;
        }

        a:hover {
            text-decoration: underline;
        }

        /* Question styling */
        h3 + blockquote {
            margin-top: 10px;
        }
    </style>
    """

    # Wrap in full HTML document
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLOps Project Report</title>
    {css}
</head>
<body>
    <div class="container">
        {html_content}
    </div>
</body>
</html>"""

    output_path = Path("reports/report.html") if Path("reports").exists() else Path("report.html")
    with output_path.open("w", encoding="utf-8") as newfile:
        newfile.write(full_html)

    logger.info(f"HTML report generated at {output_path}")


@app.command()
def check() -> None:
    """Check if report satisfies the requirements."""
    path = "reports/README.md" if Path("reports/README.md").exists() else "README.md"
    with Path(path).open() as file:
        text = file.read()

    # answers in general can be found between "Answer:" and "###" or "##"
    # which marks the next question or next section
    answers = []
    per_question = text.split("Answer:")
    per_question.pop(0)  # remove the initial section
    for q in per_question:
        if "###" in q:
            q = q.split("###")[0]
            if "##" in q:
                q = q.split("##")[0]
            answers.append(q)

    # add the last question
    answers.append(per_question[-1])

    # remove newlines
    answers = [answer.strip("\n") for answer in answers]

    question_constraints = {
        "question_1": NoConstraints(),
        "question_2": NoConstraints(),
        "question_3": LengthConstraints(min_length=0, max_length=200),
        "question_4": LengthConstraints(min_length=100, max_length=200),
        "question_5": LengthConstraints(min_length=100, max_length=200),
        "question_6": LengthConstraints(min_length=100, max_length=200),
        "question_7": LengthConstraints(min_length=50, max_length=100),
        "question_8": LengthConstraints(min_length=100, max_length=200),
        "question_9": LengthConstraints(min_length=100, max_length=200),
        "question_10": LengthConstraints(min_length=100, max_length=200),
        "question_11": LengthConstraints(min_length=200, max_length=300),
        "question_12": LengthConstraints(min_length=50, max_length=100),
        "question_13": LengthConstraints(min_length=100, max_length=200),
        "question_14": MultiConstraints(
            constrains=[
                LengthConstraints(min_length=200, max_length=300),
                ImageConstraints(min_images=1, max_images=3),
            ]
        ),
        "question_15": LengthConstraints(min_length=100, max_length=200),
        "question_16": LengthConstraints(min_length=100, max_length=200),
        "question_17": LengthConstraints(min_length=50, max_length=200),
        "question_18": LengthConstraints(min_length=100, max_length=200),
        "question_19": ImageConstraints(min_images=1, max_images=2),
        "question_20": ImageConstraints(min_images=1, max_images=2),
        "question_21": ImageConstraints(min_images=1, max_images=2),
        "question_22": LengthConstraints(min_length=100, max_length=200),
        "question_23": LengthConstraints(min_length=100, max_length=200),
        "question_24": LengthConstraints(min_length=100, max_length=200),
        "question_25": LengthConstraints(min_length=100, max_length=200),
        "question_26": LengthConstraints(min_length=100, max_length=200),
        "question_27": LengthConstraints(min_length=100, max_length=200),
        "question_28": LengthConstraints(min_length=0, max_length=200),
        "question_29": MultiConstraints(
            constrains=[
                LengthConstraints(min_length=200, max_length=400),
                ImageConstraints(min_images=1, max_images=1),
            ]
        ),
        "question_30": LengthConstraints(min_length=200, max_length=400),
        "question_31": LengthConstraints(min_length=50, max_length=300),
    }
    if len(answers) != 31:
        msg = "Number of answers are different from the expected 31. Have you changed the template?"
        raise ValueError(msg)

    counter = 0
    for index, (answer, (_, constraints)) in enumerate(zip(answers, question_constraints.items()), 1):
        counter += int(constraints(answer, index))
    logger.info(f"Total number of questions passed: {counter}/{len(answers)}")


if __name__ == "__main__":
    app()
