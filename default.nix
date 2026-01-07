{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    python311
    python311Packages.pip
    python311Packages.virtualenv
    python311Packages.streamlit
    python311Packages.pandas
    python311Packages.numpy
    python311Packages.scikit-learn
    python311Packages.matplotlib
    python311Packages.plotly
    python311Packages.pytest
    docker
    docker-compose
  ];

  shellHook = ''
    echo "Text Summarization Consistency Analyzer - Development Environment"
    echo "Python version: $(python --version)"
    echo ""
    echo "Available commands:"
    echo "  make help         - Show available make targets"
    echo "  make install      - Install dependencies"
    echo "  make test         - Run tests"
    echo "  make docker-build - Build Docker image"
    echo "  make docker-run   - Run application"
    echo ""
  '';
}
