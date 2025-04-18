# knowledge_base.py
import json
import re
import os
from typing import List, Dict, Any, Optional

class KnowledgeBase:
    """Knowledge Base for the QA Engine"""
    
    def __init__(self, kb_file_path="qa_knowledge_base.json"):
        """Initialize Knowledge Base with path to JSON file"""
        self.kb_file_path = kb_file_path
        self.qa_pairs = self._load_kb()
        
    def _load_kb(self) -> Dict[str, List]:
        """Load knowledge base from JSON file"""
        try:
            if os.path.exists(self.kb_file_path):
                with open(self.kb_file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                # Create empty knowledge base if file doesn't exist
                return {"qa_pairs": []}
        except Exception as e:
            print(f"Error loading knowledge base: {str(e)}")
            return {"qa_pairs": []}
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search knowledge base for relevant Q&A pairs
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of matching QA pairs
        """
        # Simple keyword matching
        matches = []
        query_terms = re.sub(r'[^\w\s]', '', query.lower()).split()
        
        for qa_pair in self.qa_pairs.get("qa_pairs", []):
            score = 0
            q_text = qa_pair.get("question", "").lower()
            a_text = qa_pair.get("answer", "").lower()
            category = qa_pair.get("category", "").lower()
            
            # Count matching terms in question (higher weight)
            for term in query_terms:
                if term in q_text:
                    score += 2  # Higher weight for question matches
                elif term in a_text:
                    score += 1  # Lower weight for answer matches
                elif term in category:
                    score += 0.5  # Even lower weight for category matches
            
            if score > 0:
                matches.append({
                    "qa_pair": qa_pair,
                    "score": score
                })
        
        # Sort by score and return top_k results
        matches.sort(key=lambda x: x["score"], reverse=True)
        return [m["qa_pair"] for m in matches[:top_k]]
    
    def add_qa_pair(self, question: str, answer: str, category: str = "general"):
        """
        Add a new Q&A pair to the knowledge base
        
        Args:
            question: The question
            answer: The answer
            category: Category for the Q&A pair
        """
        if "qa_pairs" not in self.qa_pairs:
            self.qa_pairs["qa_pairs"] = []
            
        self.qa_pairs["qa_pairs"].append({
            "question": question,
            "answer": answer,
            "category": category
        })
        
        # Save to file
        self._save_kb()
    
    def _save_kb(self):
        """Save knowledge base to JSON file"""
        try:
            with open(self.kb_file_path, "w", encoding="utf-8") as f:
                json.dump(self.qa_pairs, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving knowledge base: {str(e)}")
            
    def get_all_categories(self):
        """Get all unique categories in the knowledge base"""
        categories = set()
        for qa_pair in self.qa_pairs.get("qa_pairs", []):
            categories.add(qa_pair.get("category", "General"))
        return sorted(list(categories))
    
    def get_qa_pairs_by_category(self, category):
        """Get all Q&A pairs for a specific category"""
        return [qa for qa in self.qa_pairs.get("qa_pairs", []) 
                if qa.get("category", "General") == category]