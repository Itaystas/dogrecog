import { Image, StyleSheet, Platform, View, Text, Button } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

import { HelloWave } from '@/components/HelloWave';
import ParallaxScrollView from '@/components/ParallaxScrollView';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { useEffect, useState } from 'react';

export default function HomeScreen() 
{
  let [response, setResponse] = useState("NONE");
  const sendImage = (base64Data : string | undefined | null)=>
  {
      console.log(base64Data);
      fetch("https://8ac7-83-229-2-0.ngrok-free.app/hello", {
        
        method:"POST",
        headers: {
        "ngrok-skip-browser-warning": "true"
        },
        body: base64Data
      }).then(x => x.text().then(text => {setResponse(text); console.log(text)}));
  }

  const pickImage = async () => {
    // No permissions request is necessary for launching the image library
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images', 'videos'],
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
      base64: true,
    });

    console.log(result);

    if (!result.canceled) {
      sendImage(result.assets[0].base64);
    }
  };

  //useEffect(()=>{pickImage()});

  return (
    <View style={styles.container}>
      <Button title="Pick an image from camera roll" onPress={pickImage} />
      <Text style={{color:"white"}}>{response}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  image: {
    width: 200,
    height: 200,
  },
});
