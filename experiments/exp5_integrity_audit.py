import json

def run_integrity_audit():
    with open("data/retrieval_benchmark_v2.json", "r") as f:
        benchmark = json.load(f)
        
    total_queries = len(benchmark)
    total_judgments = 0
    discarded_judgments = 0
    discarded_high_rel = 0
    
    for item in benchmark:
        relevance = item.get("relevance", {})
        total_judgments += len(relevance)
        
        for doc_id_str, score in relevance.items():
            if int(doc_id_str) >= 2000:
                discarded_judgments += 1
                if score >= 3:
                    discarded_high_rel += 1
                    
    with open("experiments/results/benchmark_integrity_report.md", "w") as f:
        f.write("# Benchmark Integrity Audit\n\n")
        f.write("This report evaluates the impact of truncating the corpus to `docs[:2000]` during evaluation.\n\n")
        f.write(f"- **Total Queries:** {total_queries}\n")
        f.write(f"- **Total Relevance Judgments:** {total_judgments}\n")
        if total_judgments > 0:
            f.write(f"- **Total Judgments Discarded:** {discarded_judgments} ({(discarded_judgments/total_judgments)*100:.2f}%)\n")
        f.write(f"- **Highly Relevant (Score=3) Judgments Discarded:** {discarded_high_rel}\n")

if __name__ == "__main__":
    run_integrity_audit()
