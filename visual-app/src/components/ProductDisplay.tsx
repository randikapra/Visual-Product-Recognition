import React from "react";

type ProductDisplayProps = {
  recognizedProductImages: string[]; // An array of image URLs
  uploadedImage: string | undefined; // URL of the uploaded image
};

function ProductDisplay({
  recognizedProductImages,
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
        {recognizedProductImages.map((imageUrl, index) => (
          <div key={index} className="image-frame">
            <img
              src={imageUrl}
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
