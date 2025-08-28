#!/usr/bin/env python3
"""
Data Generation Script for Assignment 7.
This script creates sample training data for the RLHF pipeline.
"""

import os
import sys
import json
import random
from typing import List, Dict, Any
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import save_json_data


def create_training_prompts() -> List[Dict[str, str]]:
    """
    Create training prompts for RLHF.
    
    Returns:
        List of prompt dictionaries
    """
    prompts = [
        {"prompt": "How can I be more productive at work?"},
        {"prompt": "What's the best way to learn a new programming language?"},
        {"prompt": "How do I maintain a healthy work-life balance?"},
        {"prompt": "What are some effective study techniques for students?"},
        {"prompt": "How can I improve my communication skills?"},
        {"prompt": "What's the best approach to solving conflicts with colleagues?"},
        {"prompt": "How do I stay motivated when working on long-term projects?"},
        {"prompt": "What are some good habits for personal development?"},
        {"prompt": "How can I manage stress more effectively?"},
        {"prompt": "What's the best way to give constructive feedback?"},
        {"prompt": "How do I set and achieve realistic goals?"},
        {"prompt": "What are effective time management strategies?"},
        {"prompt": "How can I build better relationships with my team?"},
        {"prompt": "What's the best way to handle criticism?"},
        {"prompt": "How do I develop leadership skills?"},
        {"prompt": "What are some tips for public speaking?"},
        {"prompt": "How can I improve my problem-solving abilities?"},
        {"prompt": "What's the best way to network professionally?"},
        {"prompt": "How do I deal with difficult customers?"},
        {"prompt": "What are effective strategies for remote work?"},
        {"prompt": "How can I advance my career?"},
        {"prompt": "What's the best way to learn from failure?"},
        {"prompt": "How do I build confidence in myself?"},
        {"prompt": "What are some techniques for creative thinking?"},
        {"prompt": "How can I improve my decision-making skills?"},
        {"prompt": "What's the best approach to mentoring others?"},
        {"prompt": "How do I handle workplace pressure?"},
        {"prompt": "What are effective ways to stay organized?"},
        {"prompt": "How can I develop emotional intelligence?"},
        {"prompt": "What's the best way to adapt to change?"},
    ]
    
    return prompts


def create_evaluation_prompts() -> List[Dict[str, str]]:
    """
    Create evaluation prompts for testing.
    
    Returns:
        List of evaluation prompt dictionaries
    """
    prompts = [
        {"prompt": "How can I improve my public speaking skills?"},
        {"prompt": "What's the best way to network professionally?"},
        {"prompt": "How do I deal with difficult customers?"},
        {"prompt": "What are effective strategies for remote work?"},
        {"prompt": "How can I advance my career?"},
        {"prompt": "What's the best way to learn from mistakes?"},
        {"prompt": "How do I build team collaboration?"},
        {"prompt": "What are some techniques for innovation?"},
        {"prompt": "How can I improve my writing skills?"},
        {"prompt": "What's the best approach to project management?"},
    ]
    
    return prompts


def create_preference_data() -> List[Dict[str, str]]:
    """
    Create preference data for reward model training.
    Each entry contains a prompt, a chosen (good) response, and a rejected (bad) response.
    
    Returns:
        List of preference data dictionaries
    """
    preference_data = [
        {
            "prompt": "How can I be more productive at work?",
            "chosen": "How can I be more productive at work? Focus on prioritizing your most important tasks first, eliminate distractions by turning off notifications, take regular breaks to maintain focus, and use productivity techniques like time-blocking or the Pomodoro method. Also, organize your workspace and plan your day the night before.",
            "rejected": "How can I be more productive at work? Just work harder and longer hours until you get everything done. Don't take breaks and try to multitask as much as possible."
        },
        {
            "prompt": "What's the best way to learn a new programming language?",
            "chosen": "What's the best way to learn a new programming language? Start with the fundamentals and syntax, practice regularly with small projects, read documentation and tutorials, join online communities for support, and gradually work on more complex projects. Building real applications will help solidify your understanding.",
            "rejected": "What's the best way to learn a new programming language? Just memorize all the syntax rules and you'll be fine. Don't worry about actually writing code or understanding concepts."
        },
        {
            "prompt": "How do I maintain a healthy work-life balance?",
            "chosen": "How do I maintain a healthy work-life balance? Set clear boundaries between work and personal time, prioritize self-care activities, communicate your limits to colleagues, schedule time for hobbies and relationships, and avoid checking work emails outside of work hours. Remember that rest and recreation actually improve your work performance.",
            "rejected": "How do I maintain a healthy work-life balance? Work should always come first, so personal time isn't really important. Just focus on your career and everything else will work out."
        },
        {
            "prompt": "How can I improve my communication skills?",
            "chosen": "How can I improve my communication skills? Practice active listening by giving your full attention to others, be clear and concise in your messages, ask questions to ensure understanding, adapt your communication style to your audience, and seek feedback on your communication. Also, pay attention to non-verbal cues and body language.",
            "rejected": "How can I improve my communication skills? Just talk louder so people can hear you better. Don't worry about listening to others or being clear."
        },
        {
            "prompt": "How do I stay motivated when working on long-term projects?",
            "chosen": "How do I stay motivated when working on long-term projects? Break the project into smaller, manageable milestones, celebrate progress along the way, connect the work to your larger goals and values, maintain a support system of colleagues or mentors, and regularly review and adjust your approach as needed.",
            "rejected": "How do I stay motivated when working on long-term projects? Just force yourself to work on it even when you don't want to. Motivation isn't really necessary if you have discipline."
        },
        {
            "prompt": "What are some effective study techniques for students?",
            "chosen": "What are some effective study techniques for students? Use active recall by testing yourself regularly, employ spaced repetition to review material over time, create summary notes and mind maps, form study groups for discussion, and practice with past exams or sample problems. Also, find a quiet study environment and take regular breaks.",
            "rejected": "What are some effective study techniques for students? Just read your textbook over and over again until you memorize everything. Highlighting is enough and you don't need to actively engage with the material."
        },
        {
            "prompt": "What's the best approach to solving conflicts with colleagues?",
            "chosen": "What's the best approach to solving conflicts with colleagues? Address issues directly but respectfully, listen to understand their perspective, focus on the problem rather than personal attacks, seek common ground and mutually beneficial solutions, and involve a mediator if necessary. Always maintain professionalism throughout the process.",
            "rejected": "What's the best approach to solving conflicts with colleagues? Just avoid them completely or talk about them behind their back. Confrontation never works out well."
        },
        {
            "prompt": "What are some good habits for personal development?",
            "chosen": "What are some good habits for personal development? Read regularly to expand your knowledge, set aside time for reflection and self-assessment, seek feedback from others, practice new skills consistently, maintain a growth mindset, and invest in relationships. Also, take care of your physical and mental health.",
            "rejected": "What are some good habits for personal development? Don't worry about learning new things or changing. You're already perfect as you are and don't need to improve anything."
        },
        {
            "prompt": "How can I manage stress more effectively?",
            "chosen": "How can I manage stress more effectively? Practice stress-reduction techniques like deep breathing or meditation, maintain regular exercise and healthy eating habits, get adequate sleep, organize your tasks and priorities, seek social support when needed, and learn to say no to unnecessary commitments.",
            "rejected": "How can I manage stress more effectively? Just ignore stress and pretend it doesn't exist. Stress is just weakness and you should power through it without any coping strategies."
        },
        {
            "prompt": "What's the best way to give constructive feedback?",
            "chosen": "What's the best way to give constructive feedback? Be specific about behaviors rather than personal traits, focus on improvement opportunities, provide examples and suggestions for change, deliver feedback in private and at appropriate times, and balance criticism with recognition of strengths. Make sure your intent is to help, not to criticize.",
            "rejected": "What's the best way to give constructive feedback? Just point out everything wrong with what they did and don't worry about being specific or helpful. Criticism is always good for people."
        },
        {
            "prompt": "How do I set and achieve realistic goals?",
            "chosen": "How do I set and achieve realistic goals? Use the SMART criteria (Specific, Measurable, Achievable, Relevant, Time-bound), break large goals into smaller actionable steps, regularly track your progress, adjust your approach as needed, and celebrate milestones along the way. Also, make sure your goals align with your values.",
            "rejected": "How do I set and achieve realistic goals? Just set really easy goals that you can't fail at, or set impossible goals so you have an excuse when you don't achieve them."
        },
        {
            "prompt": "What are effective time management strategies?",
            "chosen": "What are effective time management strategies? Prioritize tasks using methods like the Eisenhower Matrix, use time-blocking to schedule focused work periods, eliminate or delegate low-value activities, batch similar tasks together, and regularly review and adjust your schedule. Also, learn to estimate how long tasks actually take.",
            "rejected": "What are effective time management strategies? Just try to do everything at once and hope for the best. Time management systems are too complicated and not worth the effort."
        },
        {
            "prompt": "How can I build better relationships with my team?",
            "chosen": "How can I build better relationships with my team? Show genuine interest in your colleagues as people, be reliable and follow through on commitments, offer help when others need it, communicate openly and honestly, and participate in team activities. Also, recognize and appreciate others' contributions.",
            "rejected": "How can I build better relationships with my team? Just focus on yourself and your own work. Building relationships is a waste of time that doesn't contribute to productivity."
        },
        {
            "prompt": "What's the best way to handle criticism?",
            "chosen": "What's the best way to handle criticism? Listen carefully without getting defensive, ask clarifying questions to understand the feedback, thank the person for their input, reflect on the validity of their points, and create an action plan for improvement. Remember that criticism can be a valuable opportunity for growth.",
            "rejected": "What's the best way to handle criticism? Get angry and defensive immediately. Anyone who criticizes you is wrong and just trying to bring you down."
        },
        {
            "prompt": "How do I develop leadership skills?",
            "chosen": "How do I develop leadership skills? Start by leading yourself effectively, seek out leadership opportunities in your current role, study successful leaders and their approaches, practice communication and decision-making skills, build emotional intelligence, and ask for feedback on your leadership style. Also, mentor others to develop your coaching abilities.",
            "rejected": "How do I develop leadership skills? Just boss people around and make all the decisions yourself. Leadership is about having power over others and making sure they do what you want."
        },
        {
            "prompt": "How can I improve my problem-solving abilities?",
            "chosen": "How can I improve my problem-solving abilities? Practice breaking complex problems into smaller components, learn different problem-solving frameworks and methodologies, seek diverse perspectives from others, analyze past solutions to understand what worked and why, and develop your critical thinking skills through practice and study.",
            "rejected": "How can I improve my problem-solving abilities? Just go with your first instinct every time. Thinking too much about problems just makes them more complicated than they need to be."
        },
        {
            "prompt": "What are effective strategies for remote work?",
            "chosen": "What are effective strategies for remote work? Establish a dedicated workspace, maintain regular work hours and routines, communicate proactively with your team, use technology tools effectively for collaboration, take regular breaks and maintain work-life boundaries, and stay connected with colleagues through virtual meetings and informal check-ins.",
            "rejected": "What are effective strategies for remote work? Just work from bed all day and don't worry about communicating with your team. Remote work means you can do whatever you want whenever you want."
        },
        {
            "prompt": "How can I advance my career?",
            "chosen": "How can I advance my career? Continuously develop your skills through learning and training, build a strong professional network, seek out challenging projects and responsibilities, find mentors and sponsors who can guide you, document and communicate your achievements, and actively plan your career path with clear goals.",
            "rejected": "How can I advance my career? Just wait for promotions to come to you automatically. Hard work will always be recognized without you having to do anything to promote yourself."
        },
        {
            "prompt": "What's the best way to learn from failure?",
            "chosen": "What's the best way to learn from failure? Analyze what went wrong without dwelling on blame, identify specific lessons and actionable insights, adjust your approach based on what you learned, maintain a growth mindset that views failure as learning opportunities, and share your experiences with others who might benefit from your insights.",
            "rejected": "What's the best way to learn from failure? Just forget about failures as quickly as possible and pretend they never happened. Thinking about failure is negative and unproductive."
        },
        {
            "prompt": "How do I build confidence in myself?",
            "chosen": "How do I build confidence in myself? Start with small achievable goals to build momentum, celebrate your successes and learn from setbacks, practice positive self-talk and challenge negative thoughts, develop your skills and competencies, seek feedback and support from others, and step outside your comfort zone gradually.",
            "rejected": "How do I build confidence in myself? Just fake it until you make it and pretend you're confident even when you're not. Real confidence isn't necessary if you can act confident."
        }
    ]
    
    return preference_data


def augment_preference_data(base_data: List[Dict]) -> List[Dict]:
    """
    Augment the preference data with additional variations.
    
    Args:
        base_data: Base preference data
        
    Returns:
        Augmented preference data
    """
    augmented_data = base_data.copy()
    
    # Add some additional challenging examples
    additional_examples = [
        {
            "prompt": "How should I handle a difficult boss?",
            "chosen": "How should I handle a difficult boss? Try to understand their communication style and preferences, document important conversations, focus on solutions rather than problems when discussing issues, maintain professionalism even when they don't, and seek support from HR or other managers if the situation becomes untenable.",
            "rejected": "How should I handle a difficult boss? Just do whatever they say without question, even if it's unreasonable. Bosses are always right and you should never push back on anything."
        },
        {
            "prompt": "What's the best way to deal with work burnout?",
            "chosen": "What's the best way to deal with work burnout? Recognize the early warning signs like exhaustion and cynicism, take time off if possible to recharge, reassess your workload and priorities, seek support from colleagues or professionals, and make changes to create better work-life balance. Prevention is better than treatment.",
            "rejected": "What's the best way to deal with work burnout? Just push through it and work even harder. Burnout is just an excuse for lazy people who can't handle pressure."
        },
        {
            "prompt": "How can I be more creative in my work?",
            "chosen": "How can I be more creative in my work? Expose yourself to diverse ideas and experiences, collaborate with people from different backgrounds, experiment with new approaches and technologies, take breaks to let your mind wander, and create an environment that encourages creative thinking and risk-taking.",
            "rejected": "How can I be more creative in my work? Just stick to the way things have always been done. Creativity is overrated and usually just causes problems and complications."
        },
        {
            "prompt": "What should I do if I made a mistake at work?",
            "chosen": "What should I do if I made a mistake at work? Take responsibility immediately and inform relevant stakeholders, assess the impact and potential solutions, work quickly to fix or mitigate the problem, learn from the mistake to prevent recurrence, and be transparent about what happened and how you're addressing it.",
            "rejected": "What should I do if I made a mistake at work? Try to hide it and hope nobody notices. If someone finds out, blame it on someone else or external circumstances."
        }
    ]
    
    augmented_data.extend(additional_examples)
    return augmented_data


def create_all_data(output_dir: str = "data"):
    """
    Create all training data files.
    
    Args:
        output_dir: Directory to save data files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔄 Creating training data...")
    
    # Create training prompts
    train_prompts = create_training_prompts()
    save_json_data(train_prompts, os.path.join(output_dir, "train_prompts.json"))
    print(f"✅ Created {len(train_prompts)} training prompts")
    
    # Create evaluation prompts
    eval_prompts = create_evaluation_prompts()
    save_json_data(eval_prompts, os.path.join(output_dir, "eval_prompts.json"))
    print(f"✅ Created {len(eval_prompts)} evaluation prompts")
    
    # Create preference data
    base_preference_data = create_preference_data()
    preference_data = augment_preference_data(base_preference_data)
    
    # Shuffle the preference data
    random.shuffle(preference_data)
    
    save_json_data(preference_data, os.path.join(output_dir, "preference_data.json"))
    print(f"✅ Created {len(preference_data)} preference pairs")
    
    # Create a sample of the data for quick testing
    sample_size = min(10, len(preference_data))
    sample_data = preference_data[:sample_size]
    save_json_data(sample_data, os.path.join(output_dir, "preference_data_sample.json"))
    print(f"✅ Created {sample_size} sample preference pairs for quick testing")
    
    print(f"\n📁 All data files saved to: {output_dir}/")
    print("📋 Files created:")
    print(f"   - train_prompts.json ({len(train_prompts)} prompts)")
    print(f"   - eval_prompts.json ({len(eval_prompts)} prompts)")
    print(f"   - preference_data.json ({len(preference_data)} pairs)")
    print(f"   - preference_data_sample.json ({sample_size} pairs)")


def validate_data(data_dir: str = "data"):
    """
    Validate the created data files.
    
    Args:
        data_dir: Directory containing data files
    """
    print("🔍 Validating data files...")
    
    files_to_check = [
        "train_prompts.json",
        "eval_prompts.json", 
        "preference_data.json"
    ]
    
    for filename in files_to_check:
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"❌ Missing file: {filepath}")
            continue
            
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if filename in ["train_prompts.json", "eval_prompts.json"]:
                # Validate prompt files
                required_keys = ["prompt"]
                for i, item in enumerate(data):
                    if not all(key in item for key in required_keys):
                        print(f"❌ {filename}: Item {i} missing required keys")
                        break
                else:
                    print(f"✅ {filename}: {len(data)} valid items")
                    
            elif filename == "preference_data.json":
                # Validate preference data
                required_keys = ["prompt", "chosen", "rejected"]
                for i, item in enumerate(data):
                    if not all(key in item for key in required_keys):
                        print(f"❌ {filename}: Item {i} missing required keys")
                        break
                else:
                    print(f"✅ {filename}: {len(data)} valid preference pairs")
                    
        except json.JSONDecodeError as e:
            print(f"❌ {filename}: Invalid JSON - {e}")
        except Exception as e:
            print(f"❌ {filename}: Error - {e}")
    
    print("✅ Data validation complete!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create training data for Assignment 7")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for data files"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing data files"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    if args.validate:
        validate_data(args.output_dir)
    else:
        create_all_data(args.output_dir)
        validate_data(args.output_dir)


if __name__ == "__main__":
    main()