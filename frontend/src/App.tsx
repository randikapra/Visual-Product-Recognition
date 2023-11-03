// App.tsx
import React, { useState } from "react";
import "./App.css";
import ImageUpload from "./components/ImageUpload";
import ProductDisplay from "./components/ProductDisplay";
import axios from "axios";
import { AxiosResponse } from "axios";

function App() {
  // const [recognizedProductImages, setRecognizedProductImages] = useState<
  //   string[]
  // >([]);

  const [similarImages, setSimilarImages] = useState<
    { image_name: string; image_base64: string }[]
  >([]);

  const [uploadedImage, setUploadedImage] = useState<string | undefined>(
    undefined
  );
  const [showAbout, setShowAbout] = useState<boolean>(false);

  const handleImageUpload = (file: File) => {
    // Set the uploaded image when handling the upload
  const imageURL = URL.createObjectURL(file);
  setUploadedImage(imageURL);

  // Send an HTTP request to your Flask backend to get similar images
  const formData = new FormData();
  formData.append('file', file);

  axios
    .post('http://127.0.0.1:5000/api/get_similar_images', formData) // Change the URL to match your Flask endpoint
    .then((response) => {
      // Assuming that the response contains 'similarImages' field with image data
      const similarImages = response.data.similarImages;
      setSimilarImages(similarImages);
    })
    .catch((error) => {
      console.error('Error fetching similar images:', error);
    }).finally(() => {
      console.log("axios")
    });

    if (showAbout) {
      setShowAbout(false); // Hide About Us when image is uploaded
    }
  };
  const handleAboutClick = () => {
    setShowAbout(!showAbout);  
  };

  return (
    <div className="App">
      <h1>Intelligence Recognizar</h1>
      <p>FASHION ECOMMERCE PLATFORM</p>
      <div className="button-container">
        <button onClick={handleAboutClick} className="about-button">
          About Us
        </button>
      </div>
      <div className="image-upload-container">
        <ImageUpload onImageUpload={handleImageUpload} />
      </div>
      <div className="product-display">
        {/* <ProductDisplay
          recognizedProductImages={recognizedProductImages}
          uploadedImage={uploadedImage}
        /> */}
        <ProductDisplay
          similarImages={similarImages}
          uploadedImage={uploadedImage}
        />
      </div>
      {showAbout && (
        <div className="about-section">
          <p>About our service:</p>
          <p>
            We provide cutting-edge visual product recognition technology that
            simplifies and enhances the shopping experience.
          </p>
          <p>
            We pride ourselves on our highly accurate prediction models. Our
            system boasts an impressive prediction accuracy, leveraging advanced
            algorithms and machine learning technology. We strive to
            consistently achieve a high accuracy rate in recognizing and
            categorizing a wide range of products, ensuring a seamless and
            efficient user experience.
          </p>
          <p>
            Our mission is to deliver top-notch services, providing not only
            accurate recognition but also a user-friendly interface. With a
            commitment to excellence, we continue to push the boundaries of
            what's possible in the realm of visual recognition technology.
          </p>
        </div>
      )}
    </div>
  );
}

export default App;
