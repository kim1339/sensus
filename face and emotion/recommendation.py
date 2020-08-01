import pandas as pd
emotion = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# makes recommendation based on user and their main detected emotion (prominent_em) 
def recommendation(name, prominent_em):
    if name == "Unknown":
        recommendations = ["User not recognized..."]
    else:
        # used pandas (pd) to read csv file of user preferences
        df = pd.read_csv("user_profiles.csv")
        
        # place the user's info in a dictionary
        user_dict = df[df['name'] == name].to_dict('records')[0]
        
        global recommendations, search
        recommendations = []
        
        # if prominent emotion is Anger, Disgust, or Fear, and user wants music, play calm genre
        if prominent_em in emotion[:3] and user_dict["suggestMusic"]:
            search = user_dict["calm genre"] + " non-copyrighted"
            recommendations.append("Personalized music/radio recommendations")
            # now use Youtube API with the "search" variable
        
        if user_dict["controlClimate"]:
            recommendations.append("Climate control (suggestions for heating, cooling, temperature adjustments)")
            
        if prominent_em in emotion[:3] and user_dict["suggestRelax"]:
            recommendations.append("Suggested physical relaxation techniques to release tension in the body (ex: unclenching steering wheel, adjusting your seat/body position, shrugging of stress in the back/shoulders/neck, etc.)")
            recommendations.append("Suggested breathing/meditation techniques")
        
        if prominent_em in emotion[:3] and user_dict["suggestBreaks"]:
            recommendations.append("Suggestions to pull over, take a break from driving, take a walk, or try to clear your head")
        
        if prominent_em in ["Sad", "Neutral"] and user_dict["suggestMusic"]:
            search = user_dict["cheer-up genre"] + " non-copyrighted"
            recommendations.append("Personalized music/radio recommendations")
            # now use Youtube API with the "search" variable

        
