import React, { useState } from "react";
import "./App.css";
import ImageUpload from "./components/ImageUpload";
import ProductDisplay from "./components/ProductDisplay";
function App() {
  const [recognizedProductImages, setRecognizedProductImages] = useState<
    string[]
  >([]);
  const [uploadedImage, setUploadedImage] = useState<string | undefined>(
    undefined
  );
  const [showAbout, setShowAbout] = useState<boolean>(false);

  const handleImageUpload = (file: File) => {
    // Set the uploaded image when handling the upload
    const imageURL = URL.createObjectURL(file);
    setUploadedImage(imageURL);

    // Simulate a backend call for recognition (replace with actual backend API call)
    // In this example, we'll set a sample array of image URLs.
    // Replace this with your actual recognition logic.
    setTimeout(() => {
      const sampleImageUrls = [
        "https://m.media-amazon.com/images/I/713R7eDlRRL._SY625_.jpg",
        "https://m.media-amazon.com/images/I/71rQFOYoszL._AC_SX569_.jpg",
        "https://img.joomcdn.net/adaebf189d5a07a001223ee6534d60a1d4053801_original.jpeg",
        "https://i5.walmartimages.com/asr/19ed609d-79ad-4ba1-89aa-ce630a808ea5.dbaed70afda332123a13742c5c5d14ee.jpeg?odnHeight=2000&odnWidth=2000&odnBg=FFFFFF",
        // "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRR84d_Za6UHPZXPe9c0fa4aPd-sxBeVYbaiA&usqp=CAU",
        // "https://assets.adidas.com/images/w_600,f_auto,q_auto/998f6eac2a1247e488f1af8f00730fca_9366/Galaxy_6_Shoes_Blue_HP2422_01_standard.jpg",
        // "https://static.nike.com/a/images/t_PDP_1280_v1/f_auto,q_auto:eco/89be1823-c476-4e99-8717-874b0e957ab2/impact-4-basketball-shoes-nmn08j.png",
        // "https://sneakerboxshop.ca/cdn/shop/products/nike-crater-impact-cw2386-101-1.jpg?v=1627074801",
        // "sample_recognized_image_9.jpg",
        // "sample_recognized_image_10.jpg",
      ];
      setRecognizedProductImages(sampleImageUrls);
    }, 2000);

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
        <ProductDisplay
          recognizedProductImages={recognizedProductImages}
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
