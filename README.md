# TitleGenerator
Sci-Ti is a creativity support system for generating titles for scientific texts.

## Installation

1. Install Python 2 (not 3).
2. Use the following command to install dependencies:
  > pip install -r requirements.txt
3. Use the following command to install TextRank (another dependency not included in requirements.txt):
  > pip install [git+git://github.com/davidadamojr/TextRank.git](git+git://github.com/davidadamojr/TextRank.git))
4. Download the stopwords and averaged_perceptron_tagger corpuses from NLTK. Instructions on how to download NLTK corpora can be found [here](https://www.nltk.org/data.html).

## Usage
Sci-Ti provides two interfaces, a graphical user interface (GUI) and a command line interface (CLI).

### GUI
1. Open the command line and change to the Sci-Ti directory.
* Run the following command:

    > python gui.py

* Sci-Ti is now running as a GUI. Click the "Select Input" button and choose a plain text file to generate titles for.
* After choosing a plain text file, Sci-Ti will automatically generate 10 titles for the contents of the file. A score will also be outputted for each title, displaying how good Sci-Ti thinks the title is relative to the other generated titles.
* To generate 10 more titles for the same file, click the "Generate Titles" button.
* By default, Sci-Ti uses a word weighting scheme to find appropriate words to use in the generated titles. To disable this weighting scheme and instead uniformly randomly select words for titles, uncheck "Use Word Weights".

### CLI
1. Open the command line and change to the Sci-Ti directory.
* Run the following command:

    > python title_generator.py

* Sci-Ti will prompt you to enter a file path (relative or absolute) of a plain text file to generate titles from. Type in a file path and press the Return/Enter key.
* Sci-Ti will output 10 titles to the console for the contents of the file. A score will also be outputted for each title, displaying how good Sci-Ti thinks the title is relative to the other generated titles.
* To generate 10 more titles, rerun the command above.
* To run Sci-Ti on a specific file immediately, without having to enter the file path every time, the "--file-path" command line argument can be specified like so:

    > python title_generator.py --file-path test.txt

* By default, Sci-Ti uses a word weighting scheme to find appropriate words to use in the generated titles. To disable this weighting scheme and instead uniformly randomly select words for titles, add the "--no-weights" flag to the command when starting Sci-Ti.
