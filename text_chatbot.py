#!/usr/bin/env python3
"""
Text-based AI Chatbot with Smart Caching and Parallel Processing
Flow: Text → Cache/Greeting Check → Quick Response + Parallel RAG → Detailed Response
Optimized for text-to-text interactions with frontend handling voice processing
"""

import logging
from typing import Optional
from dotenv import load_dotenv

from main import RAGOrchestrator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextChatbot:
    """Text-based chatbot with smart caching and parallel RAG processing."""
    
    def __init__(self):
        """Initialize the Text Chatbot."""
        print("🚀 Initializing Text Chatbot...")
        
        # Initialize the RAG orchestrator directly
        self.rag_orchestrator = RAGOrchestrator()
        self.session_id = None
        
        # Cache statistics (for tracking)
        self.cache_hits = 0
        
        print("✅ Text Chatbot initialized successfully!")
    

    
    def output_text(self, text: str) -> str:
        """Output text response (replaces speak_text for text-only mode)."""
        if not text or not text.strip():
            print("⚠️ No text to output")
            return ""
            
        print(f"🤖 RESPONSE: {text}")
        return text
    

    
    def process_text_query(self, user_query: str) -> dict:
        """Process text query and return structured response."""
        try:
            # Check for exit commands first (quick detection)
            exit_keywords = ['exit', 'quit', 'goodbye', 'bye', 'stop', 'thanks', 'thank you', 'that\'s all', 'done', 'finished']
            if any(keyword in user_query.lower() for keyword in exit_keywords):
                farewell_msg = "Thank you for using TechGropse AI Assistant. Have a great day! Goodbye!"
                return {
                    "response": farewell_msg,
                    "should_continue": False,
                    "cache_hit": False,
                    "response_type": "farewell"
                }
            
            # UNIFIED PROCESSING - Single LLM call for all analysis and quick response
            print(f"\n� Analyzing query with unified LLM call...")
            unified_analysis = self._unified_query_analysis(user_query)
            
            # Handle different analysis results
            analysis_type = unified_analysis.get("type")
            
            if analysis_type == "ending":
                print("👋 User indicated conversation ending...")
                return {
                    "response": unified_analysis.get("response"),
                    "should_continue": False,
                    "cache_hit": False,
                    "response_type": "farewell"
                }
            
            elif analysis_type == "cache_check":
                # Check cache for similar query
                cached_response = self._check_immediate_cache(user_query)
                if cached_response:
                    print("� Cache hit! Providing immediate session cached response...")
                    personalized_response = f"As I mentioned earlier in our conversation, {cached_response}"
                    print(f"🤖 Cached Response: {personalized_response}")
                    self.cache_hits += 1
                    return {
                        "response": personalized_response,
                        "should_continue": True,
                        "cache_hit": True,
                        "response_type": "cached"
                    }
                # No cache hit, continue with normal processing
                print("🔄 Cache miss. Proceeding with RAG processing...")
            
            elif analysis_type == "clarification":
                print("⚠️ Question seems incomplete or ambiguous. Asking for clarification...")
                return {
                    "response": unified_analysis.get("response"),
                    "should_continue": True,
                    "cache_hit": False,
                    "response_type": "clarification"
                }
            
            elif analysis_type == "greeting":
                print("👋 Processing greeting...")
                response = self.rag_orchestrator.process_query(
                    user_query=user_query,
                    session_id=self.session_id
                )
                
                if response and response.get("response"):
                    response_text = response["response"]
                    print(f"🤖 Greeting Response: {response_text}")
                    
                    if response.get("cache_hit"):
                        self.cache_hits += 1
                        
                    return {
                        "response": response_text,
                        "should_continue": True,
                        "cache_hit": response.get("cache_hit", False),
                        "response_type": "greeting",
                        "session_id": self.session_id
                    }
                else:
                    return {
                        "response": "Hello! How can I help you with TechGropse information today?",
                        "should_continue": True,
                        "cache_hit": False,
                        "response_type": "greeting",
                        "session_id": self.session_id
                    }
            
            # For intermediate response processing (questions that need detailed RAG)
            if unified_analysis.get("quick_response"):
                # NOT a greeting - use parallel processing with intermediate responses
                import threading
                import queue
                
                quick_response = unified_analysis.get("quick_response")
                
                # Start RAG pipeline in background thread
                response_queue = queue.Queue()
                
                def process_in_background():
                    try:
                        response = self.rag_orchestrator.process_query(
                            user_query=user_query,
                            session_id=self.session_id
                        )
                        response_queue.put(response)
                    except Exception as e:
                        logger.error(f"Background RAG processing failed: {e}")
                        response_queue.put(None)
                
                print("🚀 Starting RAG pipeline in background...")
                rag_thread = threading.Thread(target=process_in_background, daemon=True)
                rag_thread.start()
                
                # Return intermediate response first
                print(f"🚀 Quick Response: {quick_response}")
                intermediate_result = {
                    "response": quick_response,
                    "should_continue": True,
                    "cache_hit": False,
                    "response_type": "intermediate",
                    "session_id": self.session_id,
                    "has_detailed_response": True
                }
                
                # Check if RAG completed quickly (within 3 seconds)
                try:
                    response = response_queue.get(timeout=3)
                    if response and response.get("response"):
                        # RAG completed quickly - include detailed response
                        detailed_response = response["response"]
                        print(f"🤖 Detailed Response: {detailed_response}")
                        
                        # Update cache hits counter if it was a cache hit
                        if response.get("cache_hit"):
                            self.cache_hits += 1
                        
                        intermediate_result.update({
                            "detailed_response": detailed_response,
                            "cache_hit": response.get("cache_hit", False),
                            "response_type": "complete_with_intermediate"
                        })
                    else:
                        print("⚠️ Could not get detailed response quickly")
                        
                except queue.Empty:
                    # RAG still processing - will be handled later
                    print("⏳ RAG still processing - returning intermediate response")
                
                return intermediate_result
                
            else:
                # IS a greeting - process normally (no parallel processing needed)
                print("👋 Processing greeting...")
                response = self.rag_orchestrator.process_query(
                    user_query=user_query,
                    session_id=self.session_id
                )
                
                if response and response.get("response"):
                    response_text = response["response"]
                    print(f"🤖 Greeting Response: {response_text}")
                    
                    if response.get("cache_hit"):
                        self.cache_hits += 1
                        
                    return {
                        "response": response_text,
                        "should_continue": True,
                        "cache_hit": response.get("cache_hit", False),
                        "response_type": "greeting",
                        "session_id": self.session_id
                    }
                else:
                    error_msg = "Hello! How can I help you with TechGropse information today?"
                    print(f"🤖 Fallback Greeting: {error_msg}")
                    return {
                        "response": error_msg,
                        "should_continue": True,
                        "cache_hit": False,
                        "response_type": "greeting",
                        "session_id": self.session_id
                    }
            
        except Exception as e:
            logger.error(f"Text query processing failed: {e}")
            error_msg = "I apologize, but I encountered an error processing your request. Please try again."
            return {
                "response": error_msg,
                "should_continue": True,
                "cache_hit": False,
                "response_type": "error",
                "error": str(e)
            }
    
    def _check_immediate_cache(self, user_query: str) -> str:
        """
        Check session chat history for previous Q&A pairs using semantic matching with LLM.
        
        Args:
            user_query: User query string
            
        Returns:
            Cached response text from session if found, None otherwise
        """
        try:
            redis_manager = self.rag_orchestrator.redis_manager
            
            # Check session chat history for previous Q&A pairs
            session_history = redis_manager.get_chat_history(self.session_id)
            session_queries_and_responses = []
            
            if not session_history:
                print("🔍 No session history found")
                return None
                
            print(f"🔍 Checking {len(session_history)} messages in current session...")
            
            # Extract Q&A pairs from session history
            for i, message in enumerate(session_history):
                if message.get("role") == "user" and i + 1 < len(session_history):
                    user_msg = message.get("content", "")
                    next_msg = session_history[i + 1]
                    if next_msg.get("role") == "assistant":
                        assistant_response = next_msg.get("content", "")
                        if user_msg and assistant_response and len(assistant_response) > 30:
                            session_queries_and_responses.append({
                                "query": user_msg,
                                "response": assistant_response
                            })
            
            if not session_queries_and_responses:
                print("🔍 No valid Q&A pairs found in session history")
                return None
            
            # Use LLM for semantic matching within session
            print(f"🧠 Checking semantic similarity against {len(session_queries_and_responses)} session Q&A pairs...")
            
            semantic_match = self._find_semantic_match(user_query, session_queries_and_responses)
            
            if semantic_match:
                print(f"✅ Found semantic match in session history!")
                return semantic_match
            
            print("❌ No semantic match found in session history")
            return None
            
        except Exception as e:
            logger.error(f"Session cache check failed: {e}")
            return None

    def _find_semantic_match(self, user_query: str, cached_queries_and_responses: list) -> str:
        """
        Use LLM to find semantically similar cached query.
        
        Args:
            user_query: Current user query
            cached_queries_and_responses: List of cached queries and responses
            
        Returns:
            Cached response if semantic match found, None otherwise
        """
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage
            
            # Use fast model for quick semantic matching
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=100,
                timeout=3
            )
            
            # Prepare cached queries text for LLM analysis (limit to 8 for speed)
            limited_queries = cached_queries_and_responses[:8]
            cached_queries_text = ""
            for i, item in enumerate(limited_queries):
                # Truncate long queries for efficiency
                query_preview = item['query'][:100] + "..." if len(item['query']) > 100 else item['query']
                cached_queries_text += f"{i+1}. \"{query_preview}\"\n"
            
            system_prompt = f"""You are an expert at semantic similarity matching. Determine if the current user query is asking for the SAME information as any cached query.

CACHED QUERIES:
{cached_queries_text}

RULES:
- If the current query asks for the SAME core information as any cached query (even with different wording), respond with the NUMBER
- If NO semantic match exists, respond with "0"
- Be strict: only match if they want the same information

Examples:
- "What is your privacy policy?" ≈ "Tell me about privacy" ≈ "Privacy details" → MATCH
- "Hello" ≠ "What is privacy policy?" → NO MATCH"""

            user_prompt = f'Current: "{user_query}"\nResponse (number or 0):'

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = llm.invoke(messages)
            result = response.content.strip()
            
            # Parse LLM response
            try:
                match_number = int(result)
                if 1 <= match_number <= len(limited_queries):
                    matched_item = limited_queries[match_number - 1]
                    print(f"🎯 Semantic match found: \"{matched_item['query'][:60]}...\"")
                    return matched_item['response']
                elif match_number == 0:
                    return None
            except ValueError:
                print(f"⚠️ Could not parse LLM response: {result}")
                return None
            
            return None
            
        except Exception as e:
            logger.error(f"Semantic matching failed: {e}")
            return None
    
    def _check_question_completeness(self, user_query: str) -> dict:
        """
        Check if the user's question is complete and clear enough to process.
        
        Args:
            user_query: User query string
            
        Returns:
            Dict with 'is_incomplete' boolean and 'clarification_request' string
        """
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage
            
            # Use fast model for quick analysis
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3,
                max_tokens=100,
                timeout=5
            )
            
            system_prompt = """You are an AI assistant for TechGropse, a leading app development company.

Analyze if the user query is complete and clear enough to provide a meaningful response.

A query is INCOMPLETE/AMBIGUOUS if:
- Contains only partial words or fragments (like "what", "how", "child", "cookies")
- Is extremely vague without context (like "tell me about it", "what about this?")
- Has unclear pronouns without antecedents ("What about them?", "How does it work?")
- Is just a single word topic without a clear question

A query is COMPLETE if:
- Has a clear question structure ("What are the types of cookies?")
- Provides enough context to understand intent ("How do you handle privacy?")
- Is a proper greeting ("Hello", "Hi there")

RESPOND with either:
- "COMPLETE" if the query is clear enough
- "INCOMPLETE: [specific clarification request]" if it needs more information

Examples:
- "cookies" → "INCOMPLETE: Could you please clarify what you'd like to know about cookies? Are you asking about our cookie policy, types of cookies we use, or something else?"
- "what are cookies?" → "COMPLETE"
- "child" → "INCOMPLETE: I'd be happy to help with child-related information. Could you please specify what you'd like to know about children? Are you asking about child safety, privacy policies for minors, or something else?"
- "how does it work?" → "INCOMPLETE: Could you please specify what you're referring to? I'd be happy to explain how our services, policies, or features work if you could provide more context."
"""

            user_prompt = f'Query: "{user_query}"\nResponse:'

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = llm.invoke(messages)
            result = response.content.strip()
            
            if result == "COMPLETE":
                return {"is_incomplete": False, "clarification_request": None}
            elif result.startswith("INCOMPLETE:"):
                clarification = result.replace("INCOMPLETE:", "").strip()
                return {"is_incomplete": True, "clarification_request": clarification}
            else:
                # Fallback - assume complete if unclear
                return {"is_incomplete": False, "clarification_request": None}
            
        except Exception as e:
            logger.error(f"Question completeness check failed: {e}")
            # On error, assume complete to avoid blocking
            return {"is_incomplete": False, "clarification_request": None}

    def _check_and_generate_engaging_response(self, user_query: str) -> str:
        """
        Single LLM call to check if query is greeting and generate engaging response if not.
        
        Returns:
            Engaging response string if not a greeting, empty string if greeting
        """
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage
            
            # Use fast model for quick response
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=60,  # ~30-40 words for 12-15 seconds
                timeout=5
            )
            
            system_prompt = f"""You are an AI assistant for TechGropse, a leading app development company.

Analyze the user query: "{user_query}"

TASK:
1. If it's a simple greeting (hi, hello, how are you, etc.) → respond with: "GREETING"
2. If it's ending the conversation (thanks, thank you, bye, goodbye, that's all, done, finished, etc.) → respond with: "ENDING"
3. If it's NOT a greeting or ending → generate a brief engaging response (45-60 words, 12-15 seconds of speech) that:
   - Acknowledges their question warmly
   - Shows TechGropse's expertise in that area  
   - Indicates you're gathering detailed information
   - Keeps them engaged while processing

Examples:
- "hi" → "GREETING"
- "how are you" → "GREETING"
- "thanks" → "ENDING"
- "thank you" → "ENDING"
- "bye" → "ENDING"
- "that's all" → "ENDING"
- "I'm done" → "ENDING"
- "what is your privacy policy" → "Great question about privacy! At TechGropse, we prioritize data protection in all our solutions. Let me gather our comprehensive privacy details for you..."
- "child safety measures" → "Excellent inquiry about child safety! TechGropse takes child protection seriously in our app development. I'm retrieving detailed information about our safety measures..."

Be professional, conversational, and TechGropse-focused."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ]
            
            response = llm.invoke(messages)
            result = response.content.strip()
            
            # If it's a greeting, return empty string
            if result.upper() == "GREETING":
                return ""
            
            # If it's ending the conversation, return special marker
            if result.upper() == "ENDING":
                return "CONVERSATION_ENDING"
            
            # Otherwise return the engaging response
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate engaging response: {e}")
            # If LLM fails, assume it's not a greeting and provide fallback
            return "Thank you for your question! TechGropse has expertise in this area. Let me gather comprehensive information to provide you with the most accurate response."

    def create_session(self) -> str:
        """Create a new session for text-based interactions."""
        if not self.session_id:
            self.session_id = self.rag_orchestrator.create_session()
            print(f"📱 Created new session: {self.session_id}")
        return self.session_id
    
    def get_session_stats(self) -> dict:
        """Get session statistics."""
        return {
            "session_id": self.session_id,
            "cache_hits": self.cache_hits
        }
    
    def get_detailed_response(self, user_query: str, timeout: int = 30) -> dict:
        """
        Get detailed response for queries that returned intermediate responses.
        This method should be called after process_text_query returns an intermediate response.
        """
        try:
            print("🔄 Getting detailed response...")
            response = self.rag_orchestrator.process_query(
                user_query=user_query,
                session_id=self.session_id
            )
            
            if response and response.get("response"):
                response_text = response["response"]
                print(f"🤖 Detailed Response: {response_text}")
                
                # Update cache hits counter if it was a cache hit
                if response.get("cache_hit"):
                    self.cache_hits += 1
                
                return {
                    "response": response_text,
                    "cache_hit": response.get("cache_hit", False),
                    "response_type": "detailed",
                    "session_id": self.session_id
                }
            else:
                error_msg = "I apologize, but I'm having trouble accessing detailed information right now. Please try asking again."
                return {
                    "response": error_msg,
                    "cache_hit": False,
                    "response_type": "error",
                    "session_id": self.session_id
                }
                
        except Exception as e:
            logger.error(f"Failed to get detailed response: {e}")
            error_msg = "I apologize for the delay. Please try asking your question again."
            return {
                "response": error_msg,
                "cache_hit": False,
                "response_type": "error",
                "session_id": self.session_id,
                "error": str(e)
            }
    
    def _unified_query_analysis(self, user_query: str) -> dict:
        """
        Unified LLM analysis that handles greeting detection, cache checking,
        completeness validation, and quick response generation in a single call.
        
        Returns dict with analysis type and appropriate response/instructions.
        """
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage
            
            # Use fast model for quick analysis
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3,
                max_tokens=300,
                timeout=8
            )
            
            system_prompt = """You are an expert query analyzer for TechGropse AI Assistant. 

Analyze user queries and provide structured JSON responses for different query types.

ANALYSIS RULES:
1. GREETING: Simple greetings (hi, hello, hey, good morning, etc.)
2. ENDING: Conversation endings (bye, goodbye, thanks bye, farewell, etc.)  
3. CLARIFICATION: Incomplete/ambiguous queries needing more information
4. CACHE_CHECK: Complete questions that should check cache first and need detailed processing

For CACHE_CHECK queries, provide an engaging quick_response (50-70 words) that:
- Acknowledges the specific question topic
- Shows TechGropse expertise in that area
- Indicates you're gathering comprehensive information
- Keeps user engaged during processing

RESPONSE FORMAT (JSON only):
{
    "type": "greeting|ending|cache_check|clarification",
    "response": "actual response if applicable",
    "quick_response": "engaging intermediate response for cache_check queries",
    "reasoning": "brief explanation"
}"""

            user_prompt = f'''Analyze this query: "{user_query}"

Examples:
- "Hi" → {{"type": "greeting", "reasoning": "Simple greeting"}}
- "Thanks, bye!" → {{"type": "ending", "response": "Thank you for using TechGropse AI Assistant! Have a wonderful day! Goodbye!", "reasoning": "Conversation ending"}}
- "what about" → {{"type": "clarification", "response": "I'd be happy to help! Could you please specify what you'd like to know about?", "reasoning": "Incomplete query"}}
- "What are the types of cookies?" → {{"type": "cache_check", "quick_response": "Great question about cookies! TechGropse implements various cookie types for optimal user experience and website functionality. Let me gather comprehensive details about our cookie categories and their specific purposes for you...", "reasoning": "Complete query about cookies"}}

Respond with JSON only:'''

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = llm.invoke(messages)
            analysis_text = response.content.strip()
            
            # Parse JSON response
            import json
            try:
                analysis = json.loads(analysis_text)
                print(f"🧠 Unified Analysis: {analysis.get('type')} - {analysis.get('reasoning')}")
                return analysis
            except json.JSONDecodeError:
                print(f"⚠️ Failed to parse analysis JSON: {analysis_text}")
                # Fallback to basic analysis
                return self._fallback_analysis(user_query)
                
        except Exception as e:
            print(f"❌ Error in unified query analysis: {str(e)}")
            return self._fallback_analysis(user_query)
    
    def _fallback_analysis(self, user_query: str) -> dict:
        """Fallback analysis when unified LLM call fails."""
        query_lower = user_query.lower().strip()
        
        # Simple keyword-based analysis as fallback
        greeting_words = ['hi', 'hello', 'hey', 'good morning', 'good afternoon']
        ending_words = ['bye', 'goodbye', 'thanks bye', 'see you', 'farewell']
        
        if any(word in query_lower for word in greeting_words):
            return {"type": "greeting", "reasoning": "Fallback greeting detection"}
        elif any(word in query_lower for word in ending_words):
            return {
                "type": "ending", 
                "response": "Thank you for using TechGropse AI Assistant! Have a wonderful day! Goodbye!",
                "reasoning": "Fallback ending detection"
            }
        elif len(query_lower) < 10:
            return {
                "type": "clarification",
                "response": "Could you please provide more details about what you'd like to know?",
                "reasoning": "Fallback - query too short"
            }
        else:
            # Generate topic-specific quick response for common topics
            if any(word in query_lower for word in ['cookie', 'cookies']):
                quick_response = "Excellent question about cookies! TechGropse implements various cookie types for optimal user experience and website functionality. Let me gather comprehensive details about our cookie categories and their specific purposes for you..."
            elif any(word in query_lower for word in ['privacy', 'policy', 'data']):
                quick_response = "Great question about privacy! At TechGropse, we prioritize data protection and user privacy in all our solutions. Let me retrieve our comprehensive privacy policy details for you..."
            elif any(word in query_lower for word in ['service', 'services', 'what do you do']):
                quick_response = "Perfect question about our services! TechGropse offers comprehensive app development and digital solutions across multiple industries. Let me gather detailed information about our service offerings for you..."
            elif any(word in query_lower for word in ['contact', 'reach', 'location']):
                quick_response = "Great question about contacting TechGropse! We have multiple ways to connect and offices worldwide. Let me gather our current contact information and locations for you..."
            else:
                quick_response = "Thank you for your question! TechGropse has extensive expertise in this area. Let me gather comprehensive information to provide you with the most accurate and detailed response..."
            
            return {
                "type": "cache_check",
                "quick_response": quick_response,
                "reasoning": "Fallback - normal processing with topic-specific response"
            }

def main():
    """Main function for testing the text chatbot."""
    try:
        chatbot = TextChatbot()
        session_id = chatbot.create_session()
        print(f"✅ Text chatbot initialized with session: {session_id}")
        
        # Example usage
        test_query = "What is your privacy policy?"
        print(f"\n🧪 Testing with query: {test_query}")
        response = chatbot.process_text_query(test_query)
        print(f"📋 Response: {response}")
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Failed to start text chatbot: {e}")
        logger.error(f"Startup error: {e}")

if __name__ == "__main__":
    main()