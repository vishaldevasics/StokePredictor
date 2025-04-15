import mongoose, {mongo, Schema} from "mongoose";
import jwt from "jsonwebtoken" //Jwt is a bearer token.
import bcrypt from "bcrypt"


const userSchema = new Schema({
  phonenumber : {
    type : String,
    // required : true,
    lowercase : true,
    trim : true,
    index : true,
  },

  email : {
    type : String,
    required : true,
    unique : true,
    lowercase : true,
    trim : true,
  },
  
    fname : {
      type : String,
      // required : true,
      trim : true,
      index : true,
    },

    lname : {
      type : String,
      // required : true,
      trim : true,
      index : true,
    },

  password : {
    type : String,
    required : [true, 'Password is required']
  },

  refreshToken : {
    type : String,
  },
  reports: [{
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Prediction',
    default: []
  }]
   

},{
  timestamps : true
})

userSchema.pre("save", async function (next) {
  if(!this.isModified("password")) return next();

  this.password = await bcrypt.hash(this.password,10)
  next()
})

userSchema.methods.isPasswordCorrect = async function(password){
  return await bcrypt.compare(password,this.password)
}

userSchema.methods.generateAccessToken = function(){
   return jwt.sign({
    _id: this._id,
    email : this.email,
    fullname : this.fullName,
   },
   process.env.ACCESS_TOKEN_SECRET,
   {
    expiresIn : process.env.ACCESS_TOKEN_EXPIRY 
   }
  )
}
userSchema.methods.generateRefreshToken = function(){
  return jwt.sign({
    _id: this._id,
   },
   process.env.REFRESH_TOKEN_SECRET,
   {
    expiresIn : process.env.REFRESH_TOKEN_EXPIRY 
   }
  )
}

export const User = mongoose.model("User", userSchema)