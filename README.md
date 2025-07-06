

=======================================Housing Price Prediction API==============================================



**A production-ready Flask API for predicting housing prices using a trained PyTorch Neural Network model.**

---

###  **Project Highlights**

 **PyTorch Regression Model:**

* Predicts housing prices based on numerical features.
* Fully trained neural network with custom architecture.

 **Input Scaling:**

* Uses `scikit-learn` scalers for consistent preprocessing.

 **REST API:**

* Send JSON data with housing features.
* Get real-time price predictions in JSON format.

 **Postman Ready:**

* Test your API easily using Postman.

 **Deployable:**

* `Procfile` included — ready for Heroku, Railway, or any server.

---

###  **Project Structure**

```plaintext
.
├── housing_Api.py        # Flask API code
├── housing_nn.pt         # Saved PyTorch model
├── scaler_X.pkl          # Scaler for input features
├── scaler_y.pkl          # Scaler for target price
├── requirements.txt      # Project dependencies
├── Procfile              # For deploying on Heroku/Railway
└── README.md             # This documentation!
```

---

###  **How It Works**

#### 1 **Train Model (Done)**

* Model trained using PyTorch `nn.Module`.
* Architecture:

  * Input → Dense(64) → ReLU → Dense(32) → ReLU → Output(1).
* Input features: 12 housing-related numeric fields.
* `scaler_X` and `scaler_y` ensure consistent scaling.

#### 2 **API Endpoint**

* **`/predict`**

  * **Method:** `POST`
  * **Body:** JSON containing `"features"` — an array of 12 numerical values.
  * **Response:** Predicted house price.

#### 3 **Test with Postman**

* Open **Postman**.
* Make a `POST` request to: `http://127.0.0.1:5000/predict`
* Body → `raw` → `JSON`:

  ```json
  {
    "features": [0.5, 1.2, 3.3, 4.5, 2.1, 1.0, 5.6, 7.8, 0.9, 1.5, 2.3, 3.1]
  }
  ```
* Click **Send** — receive:

  ```json
  {
    "predicted_house_price": 256432.45
  }
  ```

---

###  **How to Run Locally**

####  ** 1 Clone the Repository**

```bash
git clone https://github.com/YOUR_USERNAME/housing-price-api.git
cd housing-price-api
```

####  **2 Create a Virtual Environment**

```bash
python -m venv venv
# Activate:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

####  **3- Install Dependencies**

```bash
pip install -r requirements.txt
```

####  **4- Run the API**

```bash
python housing_Api.py
```

---

###  **Deployment**

Deploy easily on:

* **Heroku**: Add `Procfile` & push.
* **Railway** / **Render**: Same process.
* **Any VPS or cloud provider**.

---

###  **What’s Included**

- Clean, tested Flask API
- Trained PyTorch model
- Scalers for input and output
- Postman-ready JSON structure
- Production-ready `requirements.txt` & `Procfile`

---

###  **Possible Improvements**

 **Accuracy Improvements:**

* Tune neural network architecture.
* Use Early Stopping in training.
* Add more training data.
* Use advanced feature engineering.

---

###  **Contact**

**Author:** Hassaan Ahmed
**Email:** hassaanahmed80400@gmail.com
**GitHub:** hassaan-ahmed825
(https://github.com/hassaan-ahmed825)



