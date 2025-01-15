# GlowAI

GlowAI is an innovative project that combines artificial intelligence with image processing to analyze your face for skin conditions such as acne, and makes recommendations for products based on the severity of the condition.

Note: Due to constraints on sourcing quality data and the cost of hosting a model in the cloud, GlowAI is currently a locally run application. It is trained on the ACNE04 dataset and is limited to identifying acne at this time.

## Features

- Comprehensive product database
- AI model trained on various skin conditions
- Intelligent product and skincare ingredient recommendations

## Installation

To install GlowAI, follow these steps:

1. Clone the repository
   ```bash
   git clone https://github.com/aa-phan/GlowAI.git
2. Navigate to the project directory:
   ```bash
   cd GlowAI
3. Install the required packages
   ```bash
   pip install -r requirements.txt

4. Make an **uploads** folder under the **static** folder
5. Follow the instructions in the training folder to build and run the Docker container, and start the model server with
   ```bash
   python endpoint.py
6. Start the application with
   ```bash
   python app.py

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Contact

For any questions or feedback, please open an issue on this repository or contact the project maintainer at atp2323@utexas.edu.

## Acknowledgements

This project was built for the Nosu AI/ML Hackathon.
https://nosu-ai-hackathon.devpost.com/

