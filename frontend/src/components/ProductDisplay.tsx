// ProductDisplay.tsx
import React from "react";

type ProductDisplayProps = {
  // recognizedProductImages: string[]; // An array of image URLs
  similarImages: { image_name: string; image_base64: string }[];
  uploadedImage: string | undefined; // URL of the uploaded image
};

function ProductDisplay({
  // recognizedProductImages,
  similarImages,
  uploadedImage,
}: ProductDisplayProps) {
  return (
    <div className="product-display">
      {uploadedImage && ( // Check if uploadedImage is available
        // <div className="image-frame">
        <div className="middle-frame">
          <p
            style={{
              fontFamily: "Times New Roman",
              fontSize: "17px",
              marginBottom: "0px",
            }}
          >
            {uploadedImage ? "Image Uploaded" : ""}
          </p>
          <img
            src={uploadedImage}
            alt="Uploaded Image"
            className="product-image"
          />
        </div>
      )}
      <div className="recognized-images">
        {/* <p
          style={{ fontFamily: "Times New Roman", fontSize: "20px" }}
        >{`Our Products: ${recognizedProductImages.length} found`}</p> */}
        {/* {recognizedProductImages.map((imageUrl, index) => (
          <div key={index} className="image-frame">
            <img
              src={imageUrl}
              alt={`Recognized Product ${index}`}
              className="product-image"
            />
          </div>
        ))} */}

        {similarImages.map((image, index) => (
          <div key={index+1} className="image-frame">
            <p>{image.image_name}</p>
            <img
              src={`data:image/jpg;base64, ${image.image_base64}`}
              alt={`Recognized Product ${index}`}
              className="product-image"
            />
          </div>
        ))}
      </div>
    </div>
  );
}

export default ProductDisplay;
