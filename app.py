import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from openai import OpenAI
from datetime import datetime
import numpy as np
from collections import defaultdict
import hashlib
import time
from dotenv import load_dotenv

# Vector store imports - using latest LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class DifficultyLevel(Enum):
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3

@dataclass
class Question:
    id: str
    text: str
    options: List[str]
    correct_answer: int
    topic: str
    subtopic: str
    difficulty: DifficultyLevel
    explanation: str
    concept_tested: str

@dataclass
class Answer:
    question_id: str
    user_answer: int
    is_correct: bool
    time_taken: float

@dataclass
class QuestionFeedback:
    is_correct: bool
    user_answer: str
    correct_answer: str
    explanation: str
    concept_clarification: str
    related_concepts: List[str]
    chapter_sections: List[Dict[str, str]]  # {"section": "1.2.1", "title": "Saving vs Investment", "page": 15}

@dataclass
class LevelAnalysis:
    level: DifficultyLevel
    total_questions: int
    correct_answers: int
    score_percentage: float
    time_taken: float
    topic_performance: Dict[str, Dict[str, float]]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    concept_gaps: List[str]
    next_level_unlocked: bool

class VectorStoreManager:
    """Manages the vector store for efficient content retrieval"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        self.vector_store = None
        
    def create_vector_store(self, chapter_content: str, chapter_num: int) -> FAISS:
        """Create vector store from chapter content"""
        # Split text into chunks
        chunks = self.text_splitter.split_text(chapter_content)
        
        # Create documents with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "chapter": chapter_num,
                    "chunk_index": i,
                    "source": f"Chapter {chapter_num}"
                }
            )
            documents.append(doc)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        return self.vector_store
    
    def search_relevant_content(self, query: str, k: int = 5) -> List[str]:
        """Search for relevant content based on query"""
        if not self.vector_store:
            return []
        
        results = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

class ChapterAnalyzer:
    """Analyzes chapter content to extract structure and topics"""
    
    def __init__(self, vector_manager: VectorStoreManager):
        self.vector_manager = vector_manager
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def extract_chapter_structure(self, chapter_content: str) -> Dict:
        """Extract topics, subtopics, and key concepts from chapter"""
        
        prompt = f"""
        Analyze this investment chapter and extract its structure.
        
        Chapter content:
        {chapter_content[:4000]}
        
        Return a JSON with:
        {{
            "main_topics": [
                {{
                    "topic": "topic name",
                    "subtopics": ["subtopic1", "subtopic2"],
                    "key_concepts": ["concept1", "concept2"],
                    "importance": "high/medium/low"
                }}
            ],
            "learning_objectives": ["objective1", "objective2"],
            "key_terms": ["term1: definition", "term2: definition"],
            "sections": [
                {{
                    "section_number": "1.1",
                    "title": "Section Title",
                    "page": 13,
                    "key_concepts": ["concept1", "concept2"],
                    "subsections": [
                        {{"number": "1.1.1", "title": "Subsection Title", "page": 13}}
                    ]
                }}
            ]
        }}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    def find_section_for_concept(self, concept: str, chapter_structure: Dict) -> List[Dict[str, str]]:
        """Find which sections discuss a particular concept"""
        sections = []
        
        for section in chapter_structure.get("sections", []):
            # Check if concept is in section's key concepts
            if concept.lower() in ' '.join(section.get("key_concepts", [])).lower():
                sections.append({
                    "section": section["section_number"],
                    "title": section["title"],
                    "page": section.get("page", 0)
                })
            
            # Check subsections
            for subsection in section.get("subsections", []):
                if concept.lower() in subsection.get("title", "").lower():
                    sections.append({
                        "section": subsection["number"],
                        "title": subsection["title"],
                        "page": subsection.get("page", 0)
                    })
        
        return sections

class AdaptiveLLMTutor:
    """Main tutor class with vector store support"""
    
    def __init__(self, chapter_content: str, chapter_num: int = 1):
        self.chapter_content = chapter_content
        self.chapter_num = chapter_num
        self.vector_manager = VectorStoreManager()
        self.chapter_analyzer = ChapterAnalyzer(self.vector_manager)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize vector store
        self.vector_store = self.vector_manager.create_vector_store(
            chapter_content, chapter_num
        )
        
        # Extract chapter structure
        self.chapter_structure = self.chapter_analyzer.extract_chapter_structure(
            chapter_content
        )
        
        # Track user performance
        self.user_responses = defaultdict(list)
        self.level_performance = {}
        self.question_cache = defaultdict(list)
        
    def generate_questions_batch(self, difficulty: DifficultyLevel, 
                                num_questions: int = 20,
                                focus_areas: Optional[List[str]] = None) -> List[Question]:
        """Generate a batch of questions based on difficulty and performance"""
        
        # Get relevant content based on difficulty
        difficulty_prompts = {
            DifficultyLevel.BEGINNER: "basic definitions, fundamental concepts, simple recall",
            DifficultyLevel.INTERMEDIATE: "application of concepts, analysis, connections between ideas",
            DifficultyLevel.ADVANCED: "complex scenarios, synthesis, evaluation, edge cases"
        }
        
        # Search for relevant content
        search_query = f"{difficulty_prompts[difficulty]} investment concepts"
        relevant_chunks = self.vector_manager.search_relevant_content(search_query, k=8)
        
        # Determine topic distribution
        topic_weights = self._calculate_topic_distribution(difficulty, focus_areas)
        
        prompt = f"""
        Generate exactly {num_questions} multiple choice questions for investment education.
        
        Chapter structure:
        {json.dumps(self.chapter_structure, indent=2)}
        
        Relevant content excerpts:
        {' '.join(relevant_chunks[:3])}
        
        Difficulty: {difficulty.name}
        Guidelines: {difficulty_prompts[difficulty]}
        
        Topic distribution (approximate):
        {json.dumps(topic_weights, indent=2)}
        
        Requirements:
        1. Each question must test a specific concept
        2. Exactly 4 options per question
        3. Clear, detailed explanations
        4. Progressive difficulty within the level
        5. Mix of question types (factual, conceptual, application)
        
        Return JSON array:
        [
            {{
                "text": "question text",
                "options": ["A", "B", "C", "D"],
                "correct_answer": 0,
                "topic": "main topic",
                "subtopic": "specific subtopic",
                "concept_tested": "specific concept being tested",
                "explanation": "why correct answer is right and others wrong",
                "difficulty_score": 1-10 within level
            }}
        ]
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        
        # Parse response and ensure it's a list
        response_content = json.loads(response.choices[0].message.content)
        if isinstance(response_content, dict) and "questions" in response_content:
            questions_data = response_content["questions"]
        elif isinstance(response_content, list):
            questions_data = response_content
        else:
            # Fallback: try to extract array from the response
            questions_data = []
        
        # Convert to Question objects
        questions = []
        for i, q_data in enumerate(questions_data[:num_questions]):
            question = Question(
                id=f"ch{self.chapter_num}_lv{difficulty.value}_q{i+1}_{int(time.time())}",
                text=q_data["text"],
                options=q_data["options"],
                correct_answer=q_data["correct_answer"],
                topic=q_data["topic"],
                subtopic=q_data.get("subtopic", ""),
                difficulty=difficulty,
                explanation=q_data["explanation"],
                concept_tested=q_data.get("concept_tested", "")
            )
            questions.append(question)
            
        # Cache questions
        cache_key = f"{difficulty.value}_{hashlib.md5(str(topic_weights).encode()).hexdigest()}"
        self.question_cache[cache_key] = questions
        
        return questions
    
    def _calculate_topic_distribution(self, difficulty: DifficultyLevel, 
                                    focus_areas: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate how many questions per topic based on performance"""
        
        topics = [topic["topic"] for topic in self.chapter_structure["main_topics"]]
        
        # Get previous level performance if exists
        if difficulty.value > 1:
            prev_level = DifficultyLevel(difficulty.value - 1)
            prev_key = f"level_{prev_level.value}"
            
            if prev_key in self.level_performance:
                perf = self.level_performance[prev_key]
                topic_perf = perf.get("topic_performance", {})
                
                weights = {}
                for topic in topics:
                    if topic in topic_perf:
                        # More questions for weaker topics
                        score = topic_perf[topic].get("score", 0.5)
                        weight = 1.5 - score
                    else:
                        weight = 1.0
                    weights[topic] = weight
                
                # Apply focus areas if specified
                if focus_areas:
                    for area in focus_areas:
                        if area in weights:
                            weights[area] *= 1.5
                
                # Normalize
                total = sum(weights.values())
                return {t: w/total for t, w in weights.items()}
        
        # Equal distribution for first attempt
        weight = 1.0 / len(topics) if topics else 1.0
        return {topic: weight for topic in topics}
    
    def evaluate_answer(self, question: Question, user_answer: int) -> QuestionFeedback:
        """Evaluate a single answer and provide immediate feedback"""
        
        is_correct = user_answer == question.correct_answer
        
        # Get additional context for explanation
        context = self.vector_manager.search_relevant_content(
            question.concept_tested, k=2
        )
        
        # Find related concepts
        related_concepts_prompt = f"""
        Given this investment concept: "{question.concept_tested}"
        And the topic: "{question.topic}"
        
        List 3-5 related concepts that the student should also understand to master this topic.
        Focus on prerequisite concepts and related ideas from the chapter.
        
        Chapter structure: {json.dumps(self.chapter_structure.get("main_topics", []), indent=2)}
        
        Return a JSON array of concepts: ["concept1", "concept2", "concept3"]
        """
        
        related_response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": related_concepts_prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        # Parse related concepts
        related_data = json.loads(related_response.choices[0].message.content)
        if isinstance(related_data, dict):
            related_concepts = related_data.get("concepts", [])
        else:
            related_concepts = []
        
        # Find relevant chapter sections
        chapter_sections = self.chapter_analyzer.find_section_for_concept(
            question.concept_tested, self.chapter_structure
        )
        
        # Also search for sections related to the topic
        topic_sections = self.chapter_analyzer.find_section_for_concept(
            question.topic, self.chapter_structure
        )
        
        # Combine and deduplicate sections
        all_sections = []
        seen = set()
        for section in chapter_sections + topic_sections:
            key = section["section"]
            if key not in seen:
                seen.add(key)
                all_sections.append(section)
        
        # Generate enhanced explanation
        enhancement_prompt = f"""
        Question: {question.text}
        User selected: {question.options[user_answer]}
        Correct answer: {question.options[question.correct_answer]}
        Basic explanation: {question.explanation}
        
        Relevant context: {context[0] if context else ""}
        
        Provide a clear, educational explanation that:
        1. Explains why the answer is {'correct' if is_correct else 'incorrect'}
        2. Clarifies the concept being tested
        3. Provides a memorable way to understand this
        4. If incorrect, explain the specific misconception
        
        Keep it concise but comprehensive (2-3 sentences).
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": enhancement_prompt}],
            temperature=0.5,
            max_tokens=300
        )
        
        enhanced_explanation = response.choices[0].message.content
        
        return QuestionFeedback(
            is_correct=is_correct,
            user_answer=question.options[user_answer],
            correct_answer=question.options[question.correct_answer],
            explanation=question.explanation,
            concept_clarification=enhanced_explanation,
            related_concepts=related_concepts[:5],  # Limit to 5
            chapter_sections=all_sections[:3]  # Limit to 3 most relevant
        )
    
    def submit_level_answers(self, difficulty: DifficultyLevel, 
                           questions: List[Question], 
                           answers: List[Answer]) -> LevelAnalysis:
        """Process all answers for a level and generate comprehensive analysis"""
        
        # Calculate performance metrics
        correct_count = sum(1 for a in answers if a.is_correct)
        total_time = sum(a.time_taken for a in answers)
        
        # Topic-wise performance
        topic_performance = defaultdict(lambda: {"correct": 0, "total": 0, "concepts": []})
        
        for question, answer in zip(questions, answers):
            topic = question.topic
            topic_performance[topic]["total"] += 1
            if answer.is_correct:
                topic_performance[topic]["correct"] += 1
            topic_performance[topic]["concepts"].append({
                "concept": question.concept_tested,
                "correct": answer.is_correct
            })
        
        # Calculate topic scores
        for topic in topic_performance:
            perf = topic_performance[topic]
            perf["score"] = perf["correct"] / perf["total"] if perf["total"] > 0 else 0
        
        # Generate analysis using LLM
        analysis_prompt = f"""
        Analyze this test performance for investment learning:
        
        Level: {difficulty.name}
        Overall: {correct_count}/{len(questions)} correct ({correct_count/len(questions)*100:.1f}%)
        Time: {total_time:.1f} seconds
        
        Topic Performance:
        {json.dumps(dict(topic_performance), indent=2)}
        
        Chapter topics available:
        {json.dumps(self.chapter_structure["main_topics"], indent=2)}
        
        Return a JSON with:
        {{
            "strengths": ["3 specific strengths shown"],
            "weaknesses": ["3 specific weaknesses"],
            "recommendations": ["4 actionable study tips"],
            "concept_gaps": ["List specific concepts to review"]
        }}
        
        Be specific about investment concepts.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.5,
            response_format={"type": "json_object"}
        )
        
        analysis_data = json.loads(response.choices[0].message.content)
        
        # Determine if next level unlocks (70% threshold)
        score_percentage = (correct_count / len(questions)) * 100
        next_level_unlocked = score_percentage >= 70
        
        # Store performance
        self.level_performance[f"level_{difficulty.value}"] = {
            "score": score_percentage,
            "topic_performance": dict(topic_performance)
        }
        
        return LevelAnalysis(
            level=difficulty,
            total_questions=len(questions),
            correct_answers=correct_count,
            score_percentage=score_percentage,
            time_taken=total_time,
            topic_performance=dict(topic_performance),
            strengths=analysis_data.get("strengths", []),
            weaknesses=analysis_data.get("weaknesses", []),
            recommendations=analysis_data.get("recommendations", []),
            concept_gaps=analysis_data.get("concept_gaps", []),
            next_level_unlocked=next_level_unlocked
        )
    
    def get_questions_json(self, difficulty: DifficultyLevel, 
                         num_questions: int = 20) -> Dict:
        """Get questions in JSON format for API response"""
        
        # Determine focus areas based on previous performance
        focus_areas = None
        if difficulty.value > 1:
            prev_level = f"level_{difficulty.value - 1}"
            if prev_level in self.level_performance:
                # Focus on weak areas
                topic_perf = self.level_performance[prev_level]["topic_performance"]
                focus_areas = [topic for topic, perf in topic_perf.items() 
                             if perf.get("score", 0) < 0.6]
        
        # Generate questions
        questions = self.generate_questions_batch(difficulty, num_questions, focus_areas)
        
        # Convert to JSON
        questions_json = []
        for q in questions:
            q_dict = asdict(q)
            q_dict['difficulty'] = q.difficulty.name
            questions_json.append(q_dict)
        
        return {
            "chapter": self.chapter_num,
            "level": difficulty.name,
            "level_number": difficulty.value,
            "questions": questions_json,
            "total_questions": len(questions_json),
            "focus_areas": focus_areas or [],
            "instructions": "Answer all questions. You'll receive immediate feedback after each answer."
        }
    
    def process_single_answer(self, question_dict: Dict, user_answer: int, 
                            time_taken: float) -> Dict:
        """Process a single answer and return immediate feedback"""
        
        # Reconstruct question object
        question = Question(
            id=question_dict['id'],
            text=question_dict['text'],
            options=question_dict['options'],
            correct_answer=question_dict['correct_answer'],
            topic=question_dict['topic'],
            subtopic=question_dict['subtopic'],
            difficulty=DifficultyLevel[question_dict['difficulty']],
            explanation=question_dict['explanation'],
            concept_tested=question_dict['concept_tested']
        )
        
        # Create answer object
        answer = Answer(
            question_id=question.id,
            user_answer=user_answer,
            is_correct=(user_answer == question.correct_answer),
            time_taken=time_taken
        )
        
        # Get feedback
        feedback = self.evaluate_answer(question, user_answer)
        
        # Store response
        self.user_responses[question.difficulty].append(answer)
        
        return {
            "question_id": question.id,
            "feedback": asdict(feedback),
            "time_taken": time_taken
        }
    
    def get_final_analysis(self, difficulty: DifficultyLevel, 
                         questions_json: List[Dict], 
                         all_answers: List[Dict]) -> Dict:
        """Generate final comprehensive analysis after all questions answered"""
        
        # Reconstruct question objects
        questions = []
        for q_data in questions_json:
            questions.append(Question(
                id=q_data['id'],
                text=q_data['text'],
                options=q_data['options'],
                correct_answer=q_data['correct_answer'],
                topic=q_data['topic'],
                subtopic=q_data['subtopic'],
                difficulty=DifficultyLevel[q_data['difficulty']],
                explanation=q_data['explanation'],
                concept_tested=q_data['concept_tested']
            ))
        
        # Create answer objects
        answers = []
        for a_data in all_answers:
            answers.append(Answer(
                question_id=a_data['question_id'],
                user_answer=a_data['user_answer'],
                is_correct=a_data['is_correct'],
                time_taken=a_data['time_taken']
            ))
        
        # Generate analysis
        analysis = self.submit_level_answers(difficulty, questions, answers)
        
        # Convert analysis to dict, handling the enum
        analysis_dict = asdict(analysis)
        analysis_dict['level'] = analysis.level.name
        
        return {
            "analysis": analysis_dict,
            "next_steps": {
                "next_level_available": analysis.next_level_unlocked,
                "recommended_focus_areas": analysis.concept_gaps[:3],
                "study_plan": analysis.recommendations
            }
        }

# API wrapper class for clean interface
class TutorAPI:
    """Clean API interface for the tutor system"""
    
    def __init__(self, chapter_content: str, chapter_num: int = 1):
        self.tutor = AdaptiveLLMTutor(chapter_content, chapter_num)
        
    def start_level(self, difficulty: str, num_questions: int = 20) -> Dict:
        """Start a new level with questions"""
        diff = DifficultyLevel[difficulty.upper()]
        return self.tutor.get_questions_json(diff, num_questions)
    
    def submit_answer(self, question: Dict, user_answer: int, time_taken: float) -> Dict:
        """Submit a single answer and get immediate feedback"""
        return self.tutor.process_single_answer(question, user_answer, time_taken)
    
    def complete_level(self, difficulty: str, questions: List[Dict], 
                      answers: List[Dict]) -> Dict:
        """Complete level and get comprehensive analysis"""
        diff = DifficultyLevel[difficulty.upper()]
        return self.tutor.get_final_analysis(diff, questions, answers)

# Example usage
if __name__ == "__main__":
    # Read chapter content from file
    with open('contents.md', 'r', encoding='utf-8') as f:
        chapter_content = f.read()
    
    # Initialize API
    api = TutorAPI(chapter_content, chapter_num=1)
    
    # 1. Start a level
    print("Starting BEGINNER level...")
    level_data = api.start_level("BEGINNER", num_questions=5)  # Using 5 for demo
    print(f"Questions generated: {len(level_data['questions'])}")
    
    # 2. Answer questions one by one (example)
    all_answers = []
    for i, question in enumerate(level_data["questions"]):
        print(f"\n--- Question {i+1} ---")
        print(f"Q: {question['text']}")
        for j, option in enumerate(question['options']):
            print(f"{j}: {option}")
        
        # Simulate user answering (for demo, using incorrect answer sometimes)
        user_answer = question['correct_answer'] if i % 2 == 0 else (question['correct_answer'] + 1) % 4
        time_taken = 45.5  # seconds
        
        # Get immediate feedback
        feedback = api.submit_answer(question, user_answer, time_taken)
        print(f"\nFeedback: {'✓ Correct!' if feedback['feedback']['is_correct'] else '✗ Incorrect'}")
        print(f"Explanation: {feedback['feedback']['concept_clarification']}")
        
        # Show related concepts to review
        if feedback['feedback'].get('related_concepts'):
            print(f"\n📚 Related Concepts to Review:")
            for concept in feedback['feedback']['related_concepts']:
                print(f"  • {concept}")
        
        # Show chapter sections to reference
        if feedback['feedback'].get('chapter_sections'):
            print(f"\n📖 Relevant Chapter Sections:")
            for section in feedback['feedback']['chapter_sections']:
                print(f"  • Section {section['section']}: {section['title']} (Page {section['page']})")
        
        # Store for final analysis
        all_answers.append({
            "question_id": question["id"],
            "user_answer": user_answer,
            "is_correct": feedback["feedback"]["is_correct"],
            "time_taken": time_taken
        })
    
    # 3. Get final analysis
    print("\n\n=== FINAL ANALYSIS ===")
    final_analysis = api.complete_level("BEGINNER", level_data["questions"], all_answers)
    print(json.dumps(final_analysis, indent=2))