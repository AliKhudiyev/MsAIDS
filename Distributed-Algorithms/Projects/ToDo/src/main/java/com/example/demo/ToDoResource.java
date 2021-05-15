package com.example.demo;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class ToDoResource {
	private ToDoRepository repo = new ToDoRepository();
	
	public static ArrayList<String> getUserNameAndPassword(Map<String, String> headers) {
		ArrayList<String> result = new ArrayList<String>();
		result.add("");
		result.add("");
		if(headers.get("username") != null) {
			result.set(0, headers.get("username"));
		}
		if(headers.get("password") != null) {
			result.set(1, headers.get("password"));
		}
		return result;
	}
	
	@GetMapping("api/todo")
	public List<ToDo> all(@RequestHeader Map<String, String> headers){
		ArrayList<String> data = ToDoResource.getUserNameAndPassword(headers);
		return repo.getToDos(data.get(0), data.get(1));
	}
	
	@GetMapping("api/users/{userId}/todo")
	public List<ToDo> all(@PathVariable Long userId, @RequestHeader Map<String, String> headers) {
		ArrayList<String> data = ToDoResource.getUserNameAndPassword(headers);
		return repo.getToDosOfUser(userId, data.get(0), data.get(1));
	}
	
	@GetMapping("api/todo/{todoId}")
	public User getOwner(@PathVariable Long todoId) {
		return repo.getOwner(todoId);
	}
	
	@GetMapping("api/users/{userId}/todo/{todoId}")
	public ToDo one(@PathVariable Long userId, @PathVariable Long todoId, @RequestHeader Map<String, String> headers) {
		ArrayList<String> data = ToDoResource.getUserNameAndPassword(headers);
		return repo.getToDoOfUser(userId, todoId, data.get(0), data.get(1));
	}
	
	@PostMapping("api/users/{userId}/todo/create/{visibility}")
	public ToDo create(@PathVariable Long userId, @PathVariable int visibility, @RequestBody ToDo todo, @RequestHeader Map<String, String> headers) {
		ArrayList<String> data = ToDoResource.getUserNameAndPassword(headers);
		todo.setVisibility(visibility == 0? false : true);
		return repo.add(userId, todo, data.get(0), data.get(1));
	}
	
	@PutMapping("api/users/{userId}/todo/update/{todoId}")
	public ToDo update(@PathVariable Long userId, @PathVariable Long todoId, @RequestBody ToDo todo, @RequestHeader Map<String, String> headers) {
		ArrayList<String> data = ToDoResource.getUserNameAndPassword(headers);
		todo.setId(todoId);
		return repo.update(userId, todo, data.get(0), data.get(1));
	}
	
	@DeleteMapping("api/users/{userId}/todo/remove/{todoId}")
	public void remove(@PathVariable Long userId, @PathVariable Long todoId, @RequestHeader Map<String, String> headers) {
		ArrayList<String> data = ToDoResource.getUserNameAndPassword(headers);
		repo.remove(userId, todoId, data.get(0), data.get(1));
	}
}
