# AI Search with Semantic Similarity

This project demonstrates how to use semantic similarity search for test descriptions using the `CrossEncoder`
from `sentence-transformers`. The code compares a query against a list of product descriptions to find the most relevant
matches, handling complexities like misspellings and variations in wording.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Contributing](#contributing)
- [Contract](#contact)
- [Credits](#credits)

## Introduction

This project leverages AI to enhance the search functionality for product descriptions. By using a
pre-trained `CrossEncoder` model, it computes the semantic similarity between a query and a list of product
descriptions, providing a robust solution for search operations.

## Features

- Semantic similarity search using `CrossEncoder`.
- Handles misspellings and variations in query terms.
- Example use case for supplier-wise product descriptions.

## Requirements

- Python 3.10
- Required packages listed in `requirements.txt`

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/itsmahadi007/Search-With-AI.git
    cd Search-With-AI
    ```

2. **Set up a virtual environment (optional but recommended):**

    ```bash
    python3.10 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare your product descriptions:**

   Update the `corpus` list with your product descriptions in the `ai_search.py` file or similar script.

2. **Run the code:**

    ```bash
    python ai_search.py
    ```

   Example of `ai_search.py`:
    ```python
    from sentence_transformers.cross_encoder import CrossEncoder
    import numpy as np

    # Load the pretrained model
    model = CrossEncoder("cross-encoder/stsb-distilroberta-base")

    # Define the query and corpus
    query = "honda 2023 fuel injectors"
    corpus = [
        "Fuel injectors for Honda Accord 2023",
        "Carburetor for 2021 Toyota Camry",
        "Engine for Ford F-150 2020",
        "Brake pads for Honda Civic 2018",
        "Fuel pump for Toyota Corolla 2019",
        "Transmission for BMW X5 2022",
        "Fuel injectors for Honda Civic 2020",
        "Alternator for Nissan Altima 2021",
        "Battery for Chevrolet Impala 2019",
    ]

    # Rank the sentences
    ranks = model.predict([[query, sentence] for sentence in corpus])
    ranked_indices = np.argsort(ranks)[::-1]

    # Print the ranked results
    print("Query: ", query)
    print("Ranked Results:")
    for idx in ranked_indices:
        print(f"{ranks[idx]:.2f}\t{corpus[idx]}")
    ```

3. **View the results:**

   The output will display the product descriptions ranked by their similarity to the query.

## Testing

1. **Install testing dependencies:**

   Testing dependencies are included in `requirements.txt`. Install them using:
    ```bash
    pip install -r requirements.txt
    ```

2. **Run tests:**

   You can run the tests using the following command:
    ```bash
    python -m unittest discover -s tests
    ```

   Ensure that your test files are placed in the `tests` directory.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## Contact

For any questions or feedback, feel free to reach out:

- Email: [mh@mahadihassan.com](mailto:mh@mahadihassan.com), [me.mahadi10@gmail.com](mailto:me.mahadi10@gmail.com)
- Github: [@itsmahadi007](https://github.com/itsmahadi007)
- Linkedin: [Mahadi Hassan](https://linkedin.com/in/mahadi-hassan-4a2239154/)
- Web: [mahadihassan.com](https://mahadihassan.com)

## Credits

This package was created by Mahadi Hassan. Special thanks to the Django and Python communities for their invaluable
resources and support.

