from datetime import datetime

class ResultsDisplayer:
    """
    A class to display and save search results.
    """

    def __init__(self, timestamp_format="%Y%m%d_%H%M%S"):
        self.timestamp_format = timestamp_format

    def display_results(self, results, keyword, query):
        """
        Displays and saves the search results.
        """
        results = sorted(results, key=lambda x: x[1], reverse=True)
        timestamp = datetime.now().strftime(self.timestamp_format)
        output_filename = f"search_results_{timestamp}.txt"
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(f"Search Results\n")
            f.write(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Full Query: {query}\n")
            f.write(f"Keyword: {keyword}\n")
            f.write(f"Total Number of Documents Found: {len(results)}\n")
            f.write("-" * 50 + "\n\n")
            
            for idx, (filename, relevance_score, similarity, tfidf, count, percentage, avg_pos) in enumerate(results):
                f.write(f"Document {idx + 1} - Title: {filename}\n")
                f.write(f"Final Relevance Score: {relevance_score:.4f}\n")
                f.write(f"Similarity Index with Query: {similarity:.4f}\n")
                f.write(f"TF-IDF for Keyword '{keyword}': {tfidf:.4f}\n")
                f.write(f"Occurrences of Keyword '{keyword}': {count}\n")
                f.write(f"Percentage of Keyword '{keyword}': {percentage:.2f}%\n")
                if avg_pos >= 0:
                    f.write(f"Average Position of Keyword '{keyword}': {avg_pos}\n")
                else:
                    f.write(f"The keyword '{keyword}' is not present in the text.\n")
                f.write("-" * 50 + "\n\n")
        
        print(f"Results have been saved in file: {output_filename}")
        for result in results:
            self._print_result(result, keyword)

    def _print_result(self, result, keyword):
        filename, relevance_score, similarity, tfidf, count, percentage, avg_pos = result
        print(f"Document - Title: {filename}")
        print(f"Final Relevance Score: {relevance_score:.4f}")
        print(f"Similarity Index with Query: {similarity:.4f}")
        print(f"TF-IDF for Keyword '{keyword}': {tfidf:.4f}")
        print(f"Occurrences of Keyword '{keyword}': {count}")
        print(f"Percentage of Keyword '{keyword}': {percentage:.2f}%")
        if avg_pos >= 0:
            print(f"Average Position of Keyword '{keyword}': {avg_pos}")
        else:
            print(f"The keyword '{keyword}' is not present in the text.")
        print("-" * 50)
