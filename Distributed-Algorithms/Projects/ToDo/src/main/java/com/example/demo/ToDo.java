package com.example.demo;

public class ToDo {
	private Long id;
	private String text;
	private boolean visibility;
	
	public Long getId() {
		return id;
	}
	
	public void setId(Long id) {
		this.id = id;
	}
	
	public String getText() {
		return text;
	}
	
	public void setText(String text) {
		this.text = text;
	}
	
	public boolean getVisibility() {
		return visibility;
	}
	
	public void setVisibility(boolean visibility) {
		this.visibility = visibility;
	}
}
