# Build with appropriate context
sudo docker build -f docker/Dockerfile.decoder --build-context lm-decoder-src=/home/raymonddunn/code/speechBCI/LanguageModelDecoder --build-context model-gru=/home/raymonddunn/data/speechBCI/ouput_dir/speechBaseline4 --build-context model-lm=/home/raymonddunn/data/speechBCI/dryad_data/languageModel -t clef-decoder .

# Run
sudo docker run --rm -p 8765:8765 clef-decoder

# Kill
sudo docker ps -q | xargs -r sudo docker kill
