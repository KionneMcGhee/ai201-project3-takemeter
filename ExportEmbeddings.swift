import Foundation
import NaturalLanguage

guard let model = NLEmbedding.wordEmbedding(for: .english) else {
    FileHandle.standardError.write(Data("English word embeddings are unavailable on this Mac.\n".utf8))
    exit(1)
}
let data = FileHandle.standardInput.readDataToEndOfFile()
let words = try JSONDecoder().decode([String].self, from: data)
let vectors = words.compactMap { word in model.vector(for: word).map { (word, $0) } }
try FileHandle.standardOutput.write(contentsOf: JSONEncoder().encode(Dictionary(uniqueKeysWithValues: vectors)))
