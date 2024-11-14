from abc import ABC, abstractmethod
import PyPDF2
import docx


class FileManager(ABC):
    @abstractmethod
    def __init__(self, file_path: str):
        self.file_path = file_path
        pass

    @abstractmethod
    def read(self) -> str:
        pass


class PDFManager(FileManager):
    def __init__(self, file_path: str):
        super().__init__(file_path)

    def read(self) -> str:
        text = ""
        try:
            with open(self.file_path, "rb") as file:
                reader = PyPDF2.PdfFileReader(file)
            for page in range(reader.numPages):
                text += reader.getPage(page).extractText() or ""
        except Exception as e:
            print(f"Error leyendo archivo PDF: {e}")
        return text


class WordManager(FileManager):
    def __init__(self, file_path: str):
        super().__init__(file_path)

    def read(self) -> str:
        text = ""
        try:
            doc = docx.Document(self.file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            print(f"Error leyendo archivo Word: {e}")
        return text


class TextManager(FileManager):
    def __init__(self, file_path: str):
        super().__init__(file_path)

    def read(self) -> str:
        text = ""
        try:
            with open(self.file_path, "r") as file:
                text = file.read()
        except Exception as e:
            print(f"Error leyendo archivo de texto: {e}")
        return text


class FileReader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_manager = self.get_file_manager()

    def get_file_manager(self) -> FileManager:
        if self.file_path.endswith(".pdf"):
            return PDFManager(self.file_path)
        elif self.file_path.endswith(".docx"):
            return WordManager(self.file_path)
        elif self.file_path.endswith(".txt"):
            return TextManager(self.file_path)
        else:
            raise ValueError("Formato de archivo no soportado.")

    def read_file(self) -> str:
        return self.file_manager.read()
